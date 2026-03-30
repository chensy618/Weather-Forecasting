"""
ERA5 Dataset Creation Utilities
================================

This module provides utility functions for creating graph-based datasets from ERA5 weather data.
It includes functions for data loading, preprocessing, graph construction, and LMDB dataset creation.

The dataset creation pipeline:
1. Load ERA5 data for a specific date
2. Apply Cartesian projection to lat/lon coordinates
3. Randomly sample spatial regions and points
4. Build k-NN graphs from the sampled points
5. Compute node features, edge attributes, and adaptive affinities
6. Store samples in LMDB format for efficient loading

Dependencies:
    - PyTorch and PyTorch Geometric
    - pandas, numpy, xarray
    - faiss (for efficient k-NN)
    - lmdb (for dataset storage)
    - pyproj (for coordinate transformations)
"""

import json
import random
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from math import pi
import calendar
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Import shared utilities from common module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from dataset_utilities import (
    setup_device,
    set_seed,
    sample_gridpoints_irregular,
    caresian_projection,
    knn_faiss,
    get_edge_indices,
    lifting,
    compute_sigmas,
    get_input_affinities,
    GraphLMDBWriter,
    GraphLMDBReader,
    load_config,
    split_train_test,
    cross_knn_faiss,
    create_fixed_grid
)

from torch_geometric.data import Data


# ============================================================================
# ERA5-specific data loading
# ============================================================================

def data_loader_new(date, time=None, vars=None, lat_max_min=None, lon_min_max=None, datafolder_path='data_hourly'):
    """Load ERA5 data from NetCDF file for a specific date."""
    file_name = os.path.join(datafolder_path, date + '.nc')
    ds = xr.open_dataset(file_name)

    # Convert all longitudes to [-180, 180] format
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

    # Apply longitude range condition
    if lon_min_max is not None:
        l0, l1 = lon_min_max
        if l0 < l1:
            ds = ds.where((ds.longitude >= l0) & (ds.longitude <= l1), drop=True)
        else:  # Handle wrapping across -180/180
            ds = ds.where((ds.longitude >= l0) | (ds.longitude <= l1), drop=True)

    # Latitude selection
    if lat_max_min is not None:
        lat0, lat1 = sorted(lat_max_min)
        ds = ds.where((ds.latitude >= lat0) & (ds.latitude <= lat1), drop=True)
    
    # Time selection
    if time is not None and 'time' in ds:
        ds = ds.sel(time=time)

    # Variable selection
    if vars is not None:
        ds = ds[vars]

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()
    df = df.rename(columns={"longitude": "lon", "latitude": "lat"})
    df = df.drop(columns=[c for c in ['valid_time', 'pressure_level', 'number', 'expver'] if c in df.columns], errors='ignore')

    return df


# ERA5-specific: normalize tensors with variable selection
def norm_tensors_from_pd(df, device):
    """Normalize tensors from pandas dataframe (ERA5 version with variable selection)."""
    # subset respective columns
    variables = [c for c in df.columns if c not in ['lat', 'lon', 'x_coord', 'y_coord', 'z_coord']]
    var_tens = torch.tensor(df.loc[:, variables].values.astype(np.float32), requires_grad=False)
    coordinate_tens = torch.tensor(df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32), requires_grad=False)
    
    # normalize variable features
    s, m = torch.std_mean(var_tens, dim=0)
    if torch.any(s == 0):
        print('Warning: std is zero for some features.')

    var_tens = (var_tens - m) / (s + 1e-6)  # add small epsilon to avoid division by zero
    var_tens = var_tens.to(device)  # to device afterwards, because 'std_mean' not implemented for mps
    
    coordinate_tens = coordinate_tens.to(device)

    # create node_feature tensor
    # Lifting disabled
    # if lifting_bool:
    #     node_tens = lifting(var_tens)
    # else:
    node_tens = var_tens

    return var_tens, coordinate_tens, node_tens


def encode_day_of_year(date_str, device):
    """Encode date as sine/cosine of day of year."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday
    days_in_year = 366 if calendar.isleap(dt.year) else 365
    angle = 2 * pi * day_of_year / days_in_year
    d1 = torch.sin(torch.tensor(angle, dtype=torch.float32, device=device))
    d2 = torch.cos(torch.tensor(angle, dtype=torch.float32, device=device))
    return d1, d2


def create_regular_grid_from_data(df, n_lat_points, n_lon_points):
    """
    Create a regular grid using actual data points.
    
    Divides the lat/lon bounding box into n_lat_points x n_lon_points cells,
    then selects the closest actual data point in each cell.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns
        n_lat_points: Number of grid points along latitude
        n_lon_points: Number of grid points along longitude
        
    Returns:
        Tuple of (grid_indices, grid_df) where:
            - grid_indices: Indices of selected points in original df
            - grid_df: DataFrame with the selected grid points
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    # Get bounds
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    # Create ideal grid
    lats_ideal = np.linspace(lat_min, lat_max, n_lat_points)
    lons_ideal = np.linspace(lon_min, lon_max, n_lon_points)
    
    # For each ideal grid point, find closest actual data point
    grid_indices = []
    for lat_target in lats_ideal:
        for lon_target in lons_ideal:
            # Find closest point in data
            distances = np.sqrt(
                (df['lat'].values - lat_target)**2 + 
                (df['lon'].values - lon_target)**2
            )
            closest_idx = np.argmin(distances)
            if closest_idx not in grid_indices:
                grid_indices.append(closest_idx)
    
    grid_indices = sorted(list(set(grid_indices)))  # Remove duplicates and sort
    grid_df = df.iloc[grid_indices].reset_index(drop=True)
    
    return grid_indices, grid_df


def create_output_node_features(coordinate_tens, date, device):
    """
    Create output graph node features using only positional and temporal encoding.
    No data variables are included.
    
    Args:
        coordinate_tens: Cartesian coordinates [N, 3]
        date: Date string for temporal encoding
        device: PyTorch device
        
    Returns:
        node_features: [N, F] tensor with positional and temporal encodings
    """
    # Normalize coordinates to unit length (positional encoding)
    coord_norm = torch.norm(coordinate_tens, dim=1, keepdim=True)
    coord_norm = torch.clamp(coord_norm, min=1e-8)
    pos_encoding = coordinate_tens / coord_norm  # [N, 3]
    
    # Temporal encoding
    d1, d2 = encode_day_of_year(date, device)
    time_encoding = torch.stack([d1, d2], dim=0).unsqueeze(0).repeat(coordinate_tens.shape[0], 1)  # [N, 2]
    
    # Concatenate positional and temporal encodings
    node_features = torch.cat([pos_encoding, time_encoding], dim=1)  # [N, 5]
    
    return node_features


def calc_edge_attr_withDateOnly(var_tens, coordinate_tens, edges, date, time):
    """Calculate edge attributes with spatial and temporal encoding (ERA5-specific)."""
    # feature diff:
    diff = []
    for f in range(var_tens.shape[1]):
        diff.append(var_tens[edges[:, 0], f] - var_tens[edges[:, 1], f])
    diff = torch.stack(diff)
    diff = diff.T
    
    # normalize
    m = torch.mean(diff, dim=0)
    s = torch.std(diff, dim=0)
    diff = (diff - m) / (s + 1e-6)  # add small epsilon to avoid division by zero

    # location main node (normalized to length one)
    loc = coordinate_tens[edges[:, 0]] / torch.norm(coordinate_tens[edges[:, 0]], dim=1).unsqueeze(1)

    # spatial distance to reference nodes (normalized vector + length)
    p1 = coordinate_tens[edges[:, 0]]
    p2 = coordinate_tens[edges[:, 1]]
    vec = p1 - p2
    dist = torch.sqrt(torch.sum((vec) ** 2, dim=1)) / 1000000  # in tsd. km
    dist = dist.unsqueeze(1)

    vec_norm = torch.norm(vec, dim=1, keepdim=True)
    vec_norm = torch.clamp(vec_norm, min=1e-8)
    vec = vec / vec_norm
    
    # Time encodings
    d1, d2 = encode_day_of_year(date, coordinate_tens.device)
    time_encoding = torch.stack([d1, d2], dim=0).unsqueeze(0).repeat(edges.shape[0], 1)

    edge_attributes = torch.cat((
        diff,  # difference in node features
        loc,  # location of main node
        vec,  # normalized vector to reference node
        dist,  # distance to reference node in tsd. km
        time_encoding
    ), dim=1)

    return edge_attributes


# ============================================================================
# High-level dataset creation functions
# ============================================================================

def prepare_dataframe(
    date: str,
    time: str,
    vars: list,
    datafolder_path: str = "data"
) -> pd.DataFrame:
    """
    Load and prepare ERA5 data for a given date.
    
    This function:
        1. Loads raw ERA5 data from NetCDF files
        2. Drops NaN values
        3. Ensures all variable columns are numeric
        4. Applies Cartesian (x, y, z) projection to lat/lon coordinates
    
    Args:
        date: Date string in YYYY-MM-DD format (e.g., "2018-01-01")
        time: Time string in HH:MM format (e.g., "00:00")
        vars: List of variable short names (e.g., ['w', 'v', 'z', 'q'])
        datafolder_path: Path to folder containing ERA5 NetCDF files
        
    Returns:
        DataFrame with columns:
            - lat, lon: Original coordinates
            - x_coord, y_coord, z_coord: Cartesian projection
            - [variable columns]: One column per variable in vars
            
    Example:
        >>> df = prepare_dataframe("2018-01-01", "00:00", ['w', 'v', 'z'])
        >>> print(df.columns)
        Index(['lat', 'lon', 'w', 'v', 'z', 'x_coord', 'y_coord', 'z_coord'])
    """
    # Load data using neural-operators utility
    df = data_loader_new(
        date,
        time=time,
        vars=vars,
        lat_max_min=None,
        lon_min_max=None,
        datafolder_path=datafolder_path
    )
    
    # Drop NaN values
    df = df.dropna()
    
    # Ensure numeric columns are properly typed
    for col in df.columns:
        if col not in ['lat', 'lon', 'date', 'time', 'pressure_level']:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
            df[col] = df[col].astype(float)
    
    # Apply Cartesian projection
    df = caresian_projection(df)
    
    return df


def create_sample(
    df: pd.DataFrame,
    date: str,
    time: str,
    n_points_min: int,
    n_points_max: int,
    k: int,
    perplexity: int,
    device: torch.device
) -> Data:
    """
    Create a single graph-structured data sample from a dataframe.
    
    This function implements the core dataset creation logic:
        1. Random spatial subsetting (lat/lon bounding box)
        2. Random point sampling within the region
        3. k-NN graph construction in Cartesian space
        4. Feature normalization and optional lifting
        5. Edge attribute computation (spatial + temporal encoding)
        6. Adaptive affinity computation using perplexity
    
    The resulting graph has:
        - Nodes: Weather observations at scattered points
        - Edges: k nearest neighbors for each node
        - Node features: Normalized atmospheric variables (optionally lifted)
        - Edge features: Spatial distances, directions, and temporal encoding
        - Input affinities: Adaptive Gaussian affinities based on feature similarity
    
    Args:
        df: Prepared dataframe with Cartesian coordinates and variables
        date: Date string for temporal encoding
        time: Time string for temporal encoding
        n_points_min: Minimum number of points to sample
        n_points_max: Maximum number of points to sample
        k: Number of nearest neighbors for graph construction
        perplexity: Perplexity parameter for affinity computation
        lifting_bool: If True, apply feature lifting (outer products)
        device: PyTorch device for computation
        
    Returns:
        PyTorch Geometric Data object with attributes:
            - x: Node features [N, D]
            - edge_index: Graph connectivity [2, N*k]
            - edge_attr: Edge features [N*k, E]
            - locations: Original lat/lon coordinates [N, 2]
            - input_affinities: Adaptive affinities [N, k]
            - date: Date string for reference
            
    Example:
        >>> df = prepare_dataframe("2018-01-01", "00:00", ['w', 'v'])
        >>> sample = create_sample(df, "2018-01-01", "00:00", 500, 2000,
        ...                         k=30, perplexity=10, lifting_bool=False,
        ...                         device=torch.device("cpu"))
        >>> print(sample.x.shape)
        torch.Size([1234, 2])  # 1234 sampled points, 2 features
    """
    # Random spatial and sample size parameters
    n_points = random.randint(n_points_min, n_points_max)
    lat_min = random.uniform(-90, 70)
    lat_max = random.uniform(lat_min + 10, 90)
    lon_min = random.uniform(-180, 160)
    lon_max = random.uniform(lon_min + 10, 180)
    
    # Spatial subsetting
    df_sample = df.loc[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    ]
    
    # Random point sampling
    if n_points < len(df_sample):
        df_sample = sample_gridpoints_irregular(df_sample, n_points)
    
    # Build k-NN graph in Cartesian space
    distances, indices = knn_faiss(
        df_sample.loc[:, ['x_coord', 'y_coord', 'z_coord']],
        k=k
    )
    edges = get_edge_indices(df_sample, indices)
    
    # Normalize tensors
    var_tens, coordinate_tens, node_tens = norm_tensors_from_pd(
        df_sample,
        device=device
    )
    
    # Edge attributes with spatial and temporal encoding
    edge_attributes = calc_edge_attr_withDateOnly(
        var_tens,
        coordinate_tens,
        edges,
        date,
        time
    )
    
    edge_index = torch.from_numpy(edges.T).detach()
    
    # Compute adaptive affinities using perplexity
    sigmas = compute_sigmas(
        node_tens,
        edge_index,
        perplexity=perplexity,
        tol=1e-5,
        max_iter=50
    )
    
    # Create PyTorch Geometric Data object
    data_sample = Data(
        x=node_tens.detach(),
        edge_index=edge_index,
        edge_attr=edge_attributes.detach(),
        locations=df_sample.loc[:, ['lat', 'lon']],
        input_affinities=get_input_affinities(var_tens, edges, sigmas=sigmas).detach(),
        date=date
    ).cpu()
    
    return data_sample


def create_interpolated_sample(
    df_full: pd.DataFrame,
    date: str,
    time: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat_out: int,
    n_lon_out: int,
    n_points_input_min: int,
    n_points_input_max: int,
    k: int,
    perplexity: int,
    device: torch.device,
    filename: str = None
) -> Data:
    """
    Create a sample with interpolated features on a regular output grid.
    
    This creates a single graph where:
    1. Output nodes are on a perfectly regular 2D grid
    2. Input nodes are randomly sampled from data
    3. Output node features are interpolated from input nodes
    4. All nodes have the same feature dimensions
    
    Args:
        df_full: Full dataframe with all data
        date: Date string
        time: Time string
        lat_min, lat_max: Latitude bounds for output grid
        lon_min, lon_max: Longitude bounds for output grid
        n_lat_out: Number of grid points along latitude
        n_lon_out: Number of grid points along longitude
        n_points_input_min: Minimum input points to sample
        n_points_input_max: Maximum input points to sample
        k: Number of neighbors for k-NN graph
        perplexity: Perplexity for affinity computation
        device: PyTorch device
        filename: Source filename
        
    Returns:
        Data object with single graph structure
    """
    from dataset_utilities import create_regular_2d_grid, interpolate_features_bilinear
    
    # Create regular output grid
    grid_df = create_regular_2d_grid(lat_min, lat_max, lon_min, lon_max, n_lat_out, n_lon_out)
    grid_df = caresian_projection(grid_df)
    
    # Sample random input points
    n_points_input = random.randint(n_points_input_min, n_points_input_max)
    n_points_input = min(n_points_input, len(df_full))
    input_indices = random.sample(range(len(df_full)), n_points_input)
    df_input = df_full.iloc[sorted(input_indices)].reset_index(drop=True)
    
    # Extract variable columns (excluding coordinates)
    var_cols = [c for c in df_input.columns if c not in ['lat', 'lon', 'x_coord', 'y_coord', 'z_coord']]
    
    # Interpolate features from input to output grid
    input_gps = df_input.loc[:, ['lat', 'lon']].values
    input_vars = df_input.loc[:, var_cols].values
    output_gps = grid_df.loc[:, ['lat', 'lon']].values
    
    output_vars = interpolate_features_bilinear(input_gps, input_vars, output_gps)
    
    # Combine input and output nodes
    combined_df = pd.concat([
        df_input,
        grid_df.assign(**{col: output_vars[:, i] for i, col in enumerate(var_cols)})
    ], ignore_index=True)
    
    # Get raw Cartesian coordinates
    coords_raw = torch.tensor(
        combined_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    # Normalize to unit sphere
    coords_normalized = coords_raw / torch.norm(coords_raw, dim=1, keepdim=True)
    
    # Build k-NN graph on combined nodes (using raw coords)
    distances, indices = knn_faiss(
        coords_raw.detach().cpu().numpy(),
        k=min(k, len(combined_df) - 1)
    )
    edges = get_edge_indices(combined_df, indices)
    
    # Normalize tensors
    var_tens, coordinate_tens, node_tens = norm_tensors_from_pd(
        combined_df,
        device=device
    )
    
    # Edge attributes
    edge_attributes = calc_edge_attr_withDateOnly(
        var_tens,
        coordinate_tens,
        edges,
        date,
        time
    )
    
    edge_index = torch.from_numpy(edges.T).long().to(device)
    
    # Compute affinities
    sigmas = compute_sigmas(
        node_tens,
        edge_index,
        perplexity=perplexity,
        tol=1e-5,
        max_iter=50
    )
    
    # GPS coordinates
    gps_coords = torch.tensor(
        combined_df.loc[:, ['lat', 'lon']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    # Create Data object
    data_sample = Data(
        x=coords_normalized.detach(),
        node_features=node_tens.detach(),
        gps=gps_coords.detach(),
        edge_index=edge_index,
        edge_attr=edge_attributes.detach(),
        input_affinities=get_input_affinities(var_tens, edges, sigmas=sigmas).detach(),
        date=date,
        filename=filename,
        n_input=n_points_input,
        n_output=len(grid_df)
    ).cpu()
    
    return data_sample


def create_lmdb_dataset(
    config: Dict[str, Any],
    dates: list,
    output_path: str,
    device: torch.device,
    time: str = "00:00",
    datafolder_path: str = "data"
) -> None:
    """
    Create a complete LMDB dataset from configuration and date list.
    
    This function orchestrates the full dataset creation pipeline:
        1. Initialize LMDB database writer
        2. For each date:
            a. Load and prepare the full ERA5 data
            b. Create multiple random samples from that data
            c. Write samples to LMDB database
        3. Close LMDB database
    
    The LMDB format provides:
        - Fast random access during training
        - Memory-mapped file I/O
        - Atomic writes and consistency
    
    Args:
        config: Configuration dictionary with dataset parameters
        dates: List of date strings to process
        output_path: Path to output LMDB file (e.g., "datasets/train.lmdb")
        device: PyTorch device to use for computation
        time: Time string for all samples (default: "00:00")
        datafolder_path: Path to folder containing ERA5 data
        
    Side Effects:
        - Creates LMDB database at output_path
        - Prints progress information to stdout
        
    Example:
        >>> config = load_config("dataset_configs/example.json")
        >>> dates = ["2018-01-01", "2018-01-02", "2018-01-03"]
        >>> create_lmdb_dataset(config, dates, "datasets/train.lmdb",
        ...                      torch.device("cpu"))
        Creating dataset: datasets/train.lmdb
        Processing 3 dates with 20 samples per date
        [1/3] 2018-01-01: 20/20 ✓
        [2/3] 2018-01-02: 20/20 ✓
        [3/3] 2018-01-03: 20/20 ✓
        ✓ Dataset creation complete: datasets/train.lmdb
    """
    # Check if dataset already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset already exists at: {output_path}")
        print(f"Please delete or rename the existing dataset before creating a new one.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Extract configuration parameters
    k = config['k']
    perplexity = config['perplexity']
    n_points_min = config['n_points_min']
    n_points_max = config['n_points_max']
    samples_per_date = config['samples_per_date']
    vars = config['vars']
    
    # Initialize LMDB writer
    writer = GraphLMDBWriter(output_path)
    
    print(f"Creating dataset: {output_path}")
    print(f"Processing {len(dates)} dates with {samples_per_date} samples per date")
    print(f"Using device: {device}\n")
    
    try:
        for date_idx, date in enumerate(dates):
            print(f"[{date_idx + 1}/{len(dates)}] {date}: ", end='')
            
            # Load and prepare data for this date
            df = prepare_dataframe(date, time, vars, datafolder_path)
            
            # Create multiple samples per date
            for sample_idx in range(samples_per_date):
                print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {sample_idx + 1}/{samples_per_date}", end='')
                
                # Create and save sample
                data_sample = create_sample(
                    df=df,
                    date=date,
                    time=time,
                    n_points_min=n_points_min,
                    n_points_max=n_points_max,
                    k=k,
                    perplexity=perplexity,
                    device=device
                )
                
                writer.append(data_sample)
            
            print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {samples_per_date}/{samples_per_date} ✓")
        
    finally:
        # Ensure LMDB is closed even if error occurs
        writer.close()
    
    print(f"\n✓ Dataset creation complete: {output_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that a configuration dictionary has all required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        KeyError: If required field is missing
        ValueError: If a field has an invalid value
        
    Example:
        >>> config = load_config("dataset_configs/example.json")
        >>> validate_config(config)  # Passes silently if valid
    """
    required_fields = [
        'k', 'perplexity', 'n_points_min', 'n_points_max',
        'samples_per_date', 'vars', 'dates'
    ]
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Required configuration field missing: {field}")
    
    # Validate value ranges
    if config['k'] <= 0:
        raise ValueError(f"k must be positive, got {config['k']}")
    if config['perplexity'] <= 0:
        raise ValueError(f"perplexity must be positive, got {config['perplexity']}")
    if config['n_points_min'] <= 0:
        raise ValueError(f"n_points_min must be positive, got {config['n_points_min']}")
    if config['n_points_max'] < config['n_points_min']:
        raise ValueError(f"n_points_max must be >= n_points_min")
    if config['samples_per_date'] <= 0:
        raise ValueError(f"samples_per_date must be positive")
    if not config['dates']:
        raise ValueError("dates list cannot be empty")


def print_dataset_summary(
    config_name: str,
    train_dates: list,
    test_dates: list,
    config: Dict[str, Any]
) -> None:
    """
    Print a summary of the dataset creation configuration.
    
    Args:
        config_name: Name of the configuration
        train_dates: List of training dates
        test_dates: List of test dates
        config: Configuration dictionary
    """
    total_dates = len(train_dates) + len(test_dates)
    samples_per_date = config['samples_per_date']
    
    print("=" * 60)
    print(f"Dataset Configuration: {config_name}")
    print("=" * 60)
    print(f"Total dates:          {total_dates}")
    print(f"  Training dates:     {len(train_dates)}")
    print(f"  Test dates:         {len(test_dates)}")
    print(f"Samples per date:     {samples_per_date}")
    print(f"  Total train samples: {len(train_dates) * samples_per_date}")
    print(f"  Total test samples:  {len(test_dates) * samples_per_date}")
    print(f"\nGraph parameters:")
    print(f"  k (neighbors):      {config['k']}")
    print(f"  Perplexity:         {config['perplexity']}")
    print(f"  Points per sample:  {config['n_points_min']}-{config['n_points_max']}")
    print(f"\nVariables: {', '.join(config['vars'])}")
    print("=" * 60 + "\n")


# ============================================================================
# Subgraph Dataset Creation (v2)
# ============================================================================

def calc_edge_attr_subgraph(coordinate_tens_in, coordinate_tens_out, cross_edges_np):
    """
    Calculate cross-edge attributes for subgraph datasets.
    
    Attributes: positional encoding, normalized direction vector, distance
    
    Args:
        coordinate_tens_in: Cartesian coordinates of input nodes [N_in, 3]
        coordinate_tens_out: Cartesian coordinates of output nodes [N_out, 3]
        cross_edges_np: Edge list [E, 2] where [0] are output indices, [1] are input indices
        
    Returns:
        edge_attributes: [E, 8] tensor with:
            - position encoding (3): normalized position of output node
            - direction vector (3): normalized (input - output) direction
            - distance (1): Euclidean distance
            - distance weight (1): exponential decay
    """
    # Position encoding: normalized position of output nodes
    pos_out = coordinate_tens_out[cross_edges_np[:, 0]]
    pos_norm = torch.norm(pos_out, dim=1, keepdim=True)
    pos_norm = torch.clamp(pos_norm, min=1e-8)
    pos_encoding = pos_out / pos_norm
    
    # Direction vector: input - output, normalized
    pos_in = coordinate_tens_in[cross_edges_np[:, 1]]
    direction = pos_in - pos_out
    dir_norm = torch.norm(direction, dim=1, keepdim=True)
    dir_norm = torch.clamp(dir_norm, min=1e-8)
    direction_normalized = direction / dir_norm
    
    # Distance and exponential weight
    distance = torch.norm(direction, dim=1).unsqueeze(1)
    distance_weight = torch.exp(-distance)
    
    edge_attributes = torch.cat([
        pos_encoding,           # [E, 3]
        direction_normalized,   # [E, 3]
        distance,              # [E, 1]
        distance_weight        # [E, 1]
    ], dim=1)
    
    return edge_attributes


def create_subgraph_sample(
    df_full: pd.DataFrame,
    date: str,
    time: str,
    n_points_input_min: int,
    n_points_input_max: int,
    grid_df: pd.DataFrame,
    k_in: int,
    k_out: int,
    k_cross: int,
    perplexity: int,
    device: torch.device,
    filename: str = None,
    cached_output: Optional[dict] = None
) -> tuple:
    """
    Create a single subgraph sample with fixed output grid and random input sampling.
    
    Args:
        df_full: Full dataframe with coordinates and features
        date: Date string for temporal encoding
        time: Time string
        n_points_input_min: Minimum input nodes
        n_points_input_max: Maximum input nodes
        n_points_output: Number of points for fixed output graph
        k_in: Neighbors within input graph
        k_out: Neighbors within output graph
        k_cross: Connections from output to input
        perplexity: Perplexity for sigma computation
        lifting_bool: Whether to apply lifting
        device: PyTorch device
        cached_output: Cached output graph data from previous sample (same date)
        
    Returns:
        Tuple of (data_sample, output_cache) where output_cache can be reused for next sample
    """
    # Output graph: use the regular grid (same for all samples)
    if cached_output is None:
        # Use the pre-defined grid
        df_output = grid_df.copy()
        
        # Raw coordinates for k-NN construction
        output_coordinates_raw = torch.tensor(
            df_output.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.astype(np.float32),
            requires_grad=False,
            device=device
        )
        
        # Normalized coordinates (unit sphere) for model input
        output_coordinates = output_coordinates_raw / torch.norm(output_coordinates_raw, dim=1, keepdim=True)
        
        var_tens_out, coordinate_tens_out, node_tens_out = norm_tensors_from_pd(
            df_output,
            device=device
        )
        
        # Create output node features using only positional and temporal encoding
        node_tens_out = create_output_node_features(
            coordinate_tens_out,
            date,
            device
        )
        
        # Build output graph edges (use raw coordinates for k-NN)
        distances_out, edge_idx_out_out = knn_faiss(
            output_coordinates_raw.detach().cpu().numpy(),
            k=min(k_out, len(df_output) - 1)
        )
        edges_out_np = get_edge_indices(df_output, edge_idx_out_out)
        edge_idx_out_out = torch.from_numpy(edges_out_np.T).long().to(device)
        
        # Compute sigmas on output graph using var_tens_out (actual features)
        # This is needed for computing input affinities later
        sigmas = compute_sigmas(
            node_tens_out,
            edge_idx_out_out,
            perplexity=perplexity,
            tol=1e-5,
            max_iter=50
        )
        
        # Output edge attributes
        edge_attr_out = calc_edge_attr_withDateOnly(
            var_tens_out,
            coordinate_tens_out,
            edges_out_np,
            date,
            time
        )
        
        # Cache output graph
        cached_output = {
            'output_coordinates': output_coordinates,
            'output_coordinates_raw': output_coordinates_raw,
            'var_tens_out': var_tens_out,
            'coordinate_tens_out': coordinate_tens_out,
            'node_tens_out': node_tens_out,
            'edge_idx_out_out': edge_idx_out_out,
            'edges_out_np': edges_out_np,
            'edge_attr_out': edge_attr_out,
            'sigmas': sigmas
        }
    else:
        # Use cached output graph
        output_coordinates = cached_output['output_coordinates']
        output_coordinates_raw = cached_output['output_coordinates_raw']
        var_tens_out = cached_output['var_tens_out']
        coordinate_tens_out = cached_output['coordinate_tens_out']
        node_tens_out = cached_output['node_tens_out']
        edge_idx_out_out = cached_output['edge_idx_out_out']
        edges_out_np = cached_output['edges_out_np']
        edge_attr_out = cached_output['edge_attr_out']
        sigmas = cached_output['sigmas']
    
    # Input graph: random subset
    n_points_input = random.randint(n_points_input_min, n_points_input_max)
    n_points_input = min(n_points_input, len(df_full))
    
    input_indices = random.sample(range(len(df_full)), n_points_input)
    input_indices_sorted = sorted(input_indices)
    
    df_input = df_full.iloc[input_indices_sorted].reset_index(drop=True)
    
    # Get raw and normalized input coordinates
    input_coordinates_raw = output_coordinates_raw[input_indices_sorted]
    input_coordinates = output_coordinates[input_indices_sorted]
    
    # Extract GPS coordinates for input and output
    input_gps = torch.tensor(
        df_input.loc[:, ['lat', 'lon']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    output_gps = torch.tensor(
        grid_df.loc[:, ['lat', 'lon']].values.astype(np.float32),
        requires_grad=False,
        device=device
    )
    
    var_tens_in, coordinate_tens_in, node_tens_in = norm_tensors_from_pd(
        df_input,
        device=device
    )
    
    # Build input graph edges (use raw coordinates for k-NN)
    distances_in, edge_idx_in_in = knn_faiss(
        input_coordinates_raw.detach().cpu().numpy(),
        k=min(k_in, n_points_input - 1)
    )
    edges_in_np = get_edge_indices(df_input, edge_idx_in_in)
    edge_idx_in_in = torch.from_numpy(edges_in_np.T).long().to(device)
    
    # Input edge attributes
    edge_attr_in = calc_edge_attr_withDateOnly(
        var_tens_in,
        coordinate_tens_in,
        edges_in_np,
        date,
        time
    )
    
    # Cross connections: output to input (use raw coordinates for k-NN)
    edge_idx_out_in = cross_knn_faiss(
        input_coordinates_raw.detach(),
        output_coordinates_raw.detach(),
        k_cross
    ).to(device)
    
    edges_cross_np = edge_idx_out_in.T.cpu().numpy()
    edge_attr_cross = calc_edge_attr_subgraph(
        coordinate_tens_in,
        coordinate_tens_out,
        edges_cross_np
    )
    
    # Create Data object
    data_sample = Data(
        # Input subgraph
        vert_in=input_coordinates.detach(),
        vert_in_features=node_tens_in.detach(),
        vert_in_gps=input_gps.detach(),
        edge_idx_in=edge_idx_in_in,
        edges_in=edge_attr_in.detach(),
        
        # Output subgraph
        vert_out=output_coordinates.detach(),
        vert_out_features=node_tens_out.detach(),
        vert_out_gps=output_gps.detach(),
        edge_idx_out=edge_idx_out_out,
        edges_out=edge_attr_out.detach(),
        
        # Cross connections
        cross_edge_idx=edge_idx_out_in,
        cross_edges=edge_attr_cross.detach(),
        
        # Additional info
        sigmas=sigmas.detach(),
        input_affinities=get_input_affinities(var_tens_out, edges_out_np, sigmas=sigmas).detach(),
        date=date,
        filename=filename
    ).cpu()
    
    return data_sample, cached_output


def create_subgraph_lmdb_dataset(
    config: Dict[str, Any],
    dates: list,
    output_path: str,
    device: torch.device,
    time: str = "00:00",
    datafolder_path: str = "data"
) -> None:
    """
    Create complete LMDB subgraph dataset.
    
    Args:
        config: Configuration with k_in, k_out, k_cross, n_points_input_min, etc.
        dates: List of date strings to process
        output_path: Path to output LMDB file
        device: PyTorch device
        time: Time string
        datafolder_path: Path to data folder
    """
    # Check if dataset already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset already exists at: {output_path}")
        print(f"Please delete or rename the existing dataset before creating a new one.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Extract configuration
    k_in = config['k_in']
    k_out = config['k_out']
    k_cross = config['k_cross']
    perplexity = config['perplexity']
    n_points_input_min = config['n_points_input_min']
    n_points_input_max = config['n_points_input_max']
    n_points_lat_out = config['n_points_lat_out']
    n_points_lon_out = config['n_points_lon_out']
    samples_per_date = config['samples_per_date']
    vars = config['vars']
    
    # Initialize writer
    writer = GraphLMDBWriter(output_path)
    
    print(f"Creating subgraph dataset: {output_path}")
    print(f"Processing {len(dates)} dates with {samples_per_date} samples per date")
    print(f"Graph structure: k_in={k_in}, k_out={k_out}, k_cross={k_cross}")
    print(f"Input nodes: {n_points_input_min}-{n_points_input_max} (random)")
    print(f"Output grid: {n_points_lat_out}x{n_points_lon_out} (fixed, reused across all samples)")
    print(f"Using device: {device}\n")
    
    # Create output grid once for entire dataset from first data file
    grid_df = None
    output_grid_created = False
    cached_output = None
    
    try:
        for date_idx, date in enumerate(dates):
            print(f"[{date_idx + 1}/{len(dates)}] {date}: ", end='')
            
            # Load and prepare data
            df = prepare_dataframe(date, time, vars, datafolder_path)
            
            # Create output grid from first date's data if not yet created
            if not output_grid_created:
                grid_indices, grid_df_temp = create_regular_grid_from_data(df, n_points_lat_out, n_points_lon_out)
                grid_df = df.iloc[grid_indices].reset_index(drop=True)
                output_grid_created = True
                print(f"Output grid created with {len(grid_df)} points\n")
            
            # Reset cached output for new date
            cached_output = None
            
            # Create samples
            for sample_idx in range(samples_per_date):
                print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {sample_idx + 1}/{samples_per_date}", end='')
                
                # Construct filename (same format as data_loader_new)
                filename = f"{date}.nc"
                
                data_sample, cached_output = create_subgraph_sample(
                    df_full=df,
                    date=date,
                    time=time,
                    n_points_input_min=n_points_input_min,
                    n_points_input_max=n_points_input_max,
                    grid_df=grid_df,
                    k_in=k_in,
                    k_out=k_out,
                    k_cross=k_cross,
                    perplexity=perplexity,
                    device=device,
                    filename=filename,
                    cached_output=cached_output
                )
                
                writer.append(data_sample)
            
            print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {samples_per_date}/{samples_per_date} ✓")
    
    finally:
        writer.close()
    
    print(f"\n✓ Subgraph dataset creation complete: {output_path}")


def validate_subgraph_config(config: Dict[str, Any]) -> None:
    """
    Validate subgraph configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        KeyError or ValueError if invalid
    """
    required_fields = [
        'k_in', 'k_out', 'k_cross', 'perplexity',
        'n_points_input_min', 'n_points_input_max', 'n_points_lat_out', 'n_points_lon_out',
        'samples_per_date', 'vars', 'dates'
    ]
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Required configuration field missing: {field}")
    
    # Validate ranges
    if config['k_in'] <= 0:
        raise ValueError(f"k_in must be positive")
    if config['k_out'] <= 0:
        raise ValueError(f"k_out must be positive")
    if config['k_cross'] <= 0:
        raise ValueError(f"k_cross must be positive")
    if config['n_points_input_min'] <= 0:
        raise ValueError(f"n_points_input_min must be positive")
    if config['n_points_input_max'] < config['n_points_input_min']:
        raise ValueError(f"n_points_input_max must be >= n_points_input_min")
    if config['n_points_lat_out'] <= 0:
        raise ValueError(f"n_points_lat_out must be positive")
    if config['n_points_lon_out'] <= 0:
        raise ValueError(f"n_points_lon_out must be positive")
