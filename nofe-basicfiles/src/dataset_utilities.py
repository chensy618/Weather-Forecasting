"""
Shared Dataset Creation Utilities
==================================

Common utilities shared between ERA5 and alphaEarth dataset creation.
This module contains core functions that are independent of data format.

Functions in this module are used by:
- ERA5/scripts/utilities_era5.py
- alphaEarth/scripts/utilities_alphaearth.py
"""

import random
import os
import pickle
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import faiss
import lmdb
from pyproj import Transformer
from torch_geometric.data import Data
from math import pi


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Device Setup
# ============================================================================

def setup_device():
    """
    Detect and return the best available PyTorch device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================================
# Sampling and Projection
# ============================================================================

def sample_gridpoints_irregular(df, N):
    """
    Randomly sample N gridpoints from dataframe.
    
    Args:
        df: DataFrame with data
        N: Number of points to sample
        
    Returns:
        Sampled dataframe
    """
    if N > len(df):
        print('N larger than available grid points, returning all points.')
        return df
    else:
        indices = list(range(len(df)))
        random.shuffle(indices)
        indices = indices[:N]
        indices.sort()
        return df.iloc[indices].reset_index(drop=True)


def caresian_projection(df, altitude=5500):
    """
    Apply Cartesian projection to lat/lon coordinates.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns
        altitude: Altitude for projection (default: 5500)
        
    Returns:
        DataFrame with added 'x_coord', 'y_coord', 'z_coord' columns
    """
    trans_GPS_to_XYZ = Transformer.from_crs(4979, 4978, always_xy=True)
    cartesian = df.apply(
        lambda row: trans_GPS_to_XYZ.transform(row['lon'], row['lat'], altitude),
        axis=1
    )
    cartesian_df = pd.DataFrame(cartesian.to_list(), columns=['x_coord', 'y_coord', 'z_coord'])
    df = pd.concat([df, cartesian_df], axis=1, ignore_index=False)
    return df


# ============================================================================
# k-NN Graph Construction
# ============================================================================

def knn_faiss(X, k=100):
    """
    KNN using faiss with L2 (Euclidean) distance.

    Args:
        X (np.ndarray or torch.Tensor): Input data of shape [N, D]
        k (int): Number of neighbors to find (excluding self)

    Returns:
        distances: [N, k] array of distances to neighbors
        indices: [N, k] array of indices of neighbors
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = X.astype('float32')

    N, D = X.shape
    index = faiss.IndexFlatL2(D)  # L2 distance index (Euclidean)
    index.add(X)

    # Search for k+1 neighbors to exclude the point itself
    distances, indices = index.search(X, k + 1)

    # Exclude self-match
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    return distances, indices


def get_edge_indices(df, indices, undirected=False):
    """
    Construct edge list from k-NN indices.
    
    Args:
        df: DataFrame (used for shape)
        indices: k-NN indices array
        undirected: If True, add reverse edges
        
    Returns:
        Edge array of shape [E, 2]
    """
    node_indices = np.arange(df.shape[0])
    edges = np.column_stack((np.repeat(node_indices, indices.shape[1]), indices.flatten()))
    if undirected:
        reverse_edges = np.column_stack((edges[:, 1], edges[:, 0]))
        edges = np.vstack((edges, reverse_edges))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
    return edges


def cross_knn_faiss(X, Y, k):
    """
    Find k nearest neighbors from X for each point in Y using faiss.
    
    Args:
        X (np.ndarray or torch.Tensor): Source points [N, D]
        Y (np.ndarray or torch.Tensor): Query points [M, D]
        k (int): Number of neighbors to find
        
    Returns:
        edge_index: [2, M*k] tensor where [0,:] are Y indices and [1,:] are X indices
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    X = X.astype('float32')
    Y = Y.astype('float32')
    
    N_x, D = X.shape
    N_y = Y.shape[0]
    
    # Build index on source points (X)
    index = faiss.IndexFlatL2(D)
    index.add(X)
    
    # Query for k nearest neighbors from X for each point in Y
    distances, indices = index.search(Y, k)
    
    # Build edge index: [0,:] = Y indices (repeated), [1,:] = X indices (neighbors)
    y_indices = np.repeat(np.arange(N_y), k)
    x_indices = indices.flatten()
    
    edge_index = torch.from_numpy(np.stack([y_indices, x_indices], axis=0)).long()
    
    return edge_index


def create_fixed_grid(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    """
    Create a fixed regular grid of points.
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        lat_step, lon_step: Grid spacing
        
    Returns:
        DataFrame with lat, lon columns for grid points
    """
    lats = np.arange(lat_min, lat_max + lat_step/2, lat_step)
    lons = np.arange(lon_min, lon_max + lon_step/2, lon_step)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    df = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten()
    })
    
    return df


def create_regular_2d_grid(lat_min, lat_max, lon_min, lon_max, n_lat, n_lon):
    """
    Create a truly regular 2D grid using np.linspace.
    
    Unlike create_regular_grid_from_data which picks nearest existing points,
    this creates a perfectly regular grid within the specified bounds.
    
    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        n_lat: Number of points along latitude
        n_lon: Number of points along longitude
        
    Returns:
        DataFrame with lat, lon columns for grid points
    """
    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)
    
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    df = pd.DataFrame({
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten()
    })
    
    return df


def interpolate_features_bilinear(input_gps, input_features, output_gps):
    """
    Interpolate features from input points to output points using bilinear interpolation.
    
    Uses scipy's griddata with linear interpolation for scattered data.
    Points outside the convex hull use nearest neighbor extrapolation.
    
    Args:
        input_gps: [N_in, 2] array of (lat, lon) for input points
        input_features: [N_in, D] array of features at input points
        output_gps: [N_out, 2] array of (lat, lon) for output points
        
    Returns:
        output_features: [N_out, D] array of interpolated features
    """
    from scipy.interpolate import griddata
    
    # Convert to numpy if needed
    if isinstance(input_gps, torch.Tensor):
        input_gps = input_gps.cpu().numpy()
    if isinstance(input_features, torch.Tensor):
        input_features = input_features.cpu().numpy()
    if isinstance(output_gps, torch.Tensor):
        output_gps = output_gps.cpu().numpy()
    
    # Interpolate each feature dimension
    output_features = griddata(
        points=input_gps,
        values=input_features,
        xi=output_gps,
        method='linear',
        fill_value=np.nan
    )
    
    # For points outside convex hull (NaN), use nearest neighbor
    nan_mask = np.isnan(output_features).any(axis=1)
    if nan_mask.any():
        nearest_features = griddata(
            points=input_gps,
            values=input_features,
            xi=output_gps[nan_mask],
            method='nearest'
        )
        output_features[nan_mask] = nearest_features
    
    return output_features


# ============================================================================
# Feature Processing
# ============================================================================

def lifting(data_in):
    """
    Apply feature lifting (outer products).
    
    Args:
        data_in: Input tensor [N, D]
        
    Returns:
        Lifted features [N, D + D^2]
    """
    l = data_in.shape[0]
    outer_products = torch.bmm(data_in.unsqueeze(2), data_in.unsqueeze(1))
    outer_products_flat = outer_products.view(l, -1)
    result = torch.cat((data_in, outer_products_flat), dim=1)
    return result


# ============================================================================
# Perplexity-based Affinities
# ============================================================================

def compute_sigmas(X, edge_index, perplexity, tol=1e-5, max_iter=50):
    """
    Compute per-point sigma values to match a target perplexity.
    
    Uses binary search to find sigma for each point such that the
    entropy of its neighbor distribution matches the target perplexity.
    
    Args:
        X (Tensor): [N, D] data points
        edge_index (LongTensor): [2, N * k] edge indices
        perplexity (float): Target perplexity
        tol (float): Tolerance for convergence
        max_iter (int): Maximum iterations
        
    Returns:
        sigmas (Tensor): [N] sigma values per point
    """
    device = X.device
    N = X.size(0)
    k = edge_index.size(1) // N

    src = edge_index[0]
    tgt = edge_index[1]

    # Pairwise squared distances between neighbors
    xi = X[src]
    xj = X[tgt]
    dists = torch.sum((xi - xj) ** 2, dim=1)
    dists = dists.view(N, k)
    dists = torch.clamp(dists, min=1e-8)

    log_perp = torch.log(torch.tensor(perplexity, device=device))

    # Binary search parameters
    beta = torch.ones(N, device=device)
    beta_min = torch.full((N,), -float("inf"), device=device)
    beta_max = torch.full((N,), float("inf"), device=device)

    for _ in range(max_iter):
        P = torch.exp(-dists * beta[:, None])
        sumP = P.sum(dim=1, keepdim=True)
        sumP = torch.clamp(sumP, min=1e-8)
        P = P / sumP

        entropy = -torch.sum(P * torch.log(P + 1e-10), dim=1)
        H_diff = entropy - log_perp

        mask = H_diff.abs() > tol

        increase = H_diff > tol
        decrease = ~increase

        beta_min[increase] = beta[increase]
        beta_max[decrease] = beta[decrease]

        beta[increase] = torch.where(
            beta_max[increase] == float("inf"),
            beta[increase] * 2,
            (beta[increase] + beta_max[increase]) / 2
        )

        beta[decrease] = torch.where(
            beta_min[decrease] == -float("inf"),
            beta[decrease] / 2,
            (beta[decrease] + beta_min[decrease]) / 2
        )

        if not mask.any():
            break

    # Final sigma
    sigmas = torch.sqrt(1.0 / (2.0 * beta))
    return sigmas


def get_input_affinities(var_tens, edges, sigmas):
    """
    Compute local affinity matrix using per-point sigma values.

    Args:
        var_tens (Tensor): [N, D] input features
        edges (LongTensor): [E, 2] edge list
        sigmas (Tensor): [N] per-point sigma values

    Returns:
        Tensor: [N, k] affinity values (p_j|i) normalized per source node
    """
    eps = 1e-6

    x_i = var_tens[edges[:, 0]]
    x_j = var_tens[edges[:, 1]]
    sigma_i = sigmas[edges[:, 0]]

    sqdist = torch.sum((x_i - x_j) ** 2, dim=1)

    denom = 2 * sigma_i ** 2 + eps
    v = torch.exp(-sqdist / denom)

    N = var_tens.shape[0]
    k = v.numel() // N
    v = v.view(N, k)

    p = v / (v.sum(dim=1, keepdim=True) + eps)

    return p


# ============================================================================
# LMDB Storage
# ============================================================================

class GraphLMDBWriter:
    """
    Writer for storing graph data in LMDB format.
    
    Usage:
        writer = GraphLMDBWriter('dataset.lmdb')
        writer.append(data_sample)
        writer.close()
    """
    
    def __init__(self, db_path, map_size=int(1e12)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)
        
        with self.env.begin(write=True) as txn:
            raw_len = txn.get(b'__len__')
            self.length = int(raw_len.decode()) if raw_len else 0

    def append(self, data: Data, to_cpu=True):
        if to_cpu:
            data = data.detach().cpu()
        with self.env.begin(write=True) as txn:
            txn.put(str(self.length).encode(), pickle.dumps(data))
            self.length += 1
            txn.put(b'__len__', str(self.length).encode())

    def close(self):
        self.env.close()


class GraphLMDBReader:
    """
    Reader for loading graph data from LMDB databases.
    
    Usage:
        reader = GraphLMDBReader('dataset.lmdb')
        sample = reader[0]
    """
    
    def __init__(self, db_path):
        self.env = lmdb.open(db_path, readonly=True, lock=False, map_size=10 * 1024**3)
        with self.env.begin() as txn:
            self.length = int(txn.get(b'__len__').decode())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            byte_data = txn.get(str(idx).encode())
            return pickle.loads(byte_data)


# ============================================================================
# Configuration and Validation
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load dataset configuration from JSON file."""
    import json
    with open(config_path, "r") as f:
        return json.load(f)


def split_train_test(
    items: list,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[list, list]:
    """
    Split list into training and test sets.
    
    Args:
        items: List of items (dates, years, etc.)
        train_ratio: Fraction for training
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_items, test_items)
    """
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    items_copy = items.copy()
    
    if shuffle:
        random.seed(random_seed)
        random.shuffle(items_copy)
    
    split_idx = int(len(items_copy) * train_ratio)
    train_items = items_copy[:split_idx]
    test_items = items_copy[split_idx:]
    
    return train_items, test_items
