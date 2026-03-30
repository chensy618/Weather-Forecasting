#!/usr/bin/env python3
"""
Interpolated Dataset Creation Script for ERA5 Weather Data
===========================================================

Creates train and test LMDB datasets with interpolated features on regular grids.

This workflow:
1. Creates perfectly regular 2D output grids using np.linspace
2. Samples random input points from data
3. Interpolates features from input to output nodes
4. Creates single graphs with uniform node features (compatible with regular KernelNN)

Usage:
    python create_interpolated_dataset.py <config_name> [--no-test] [--data_path PATH]
    
    config_name: Name of config file in dataset_configs_interpolated/ (without .json extension)
    --no-test: Optional flag to skip test dataset creation
    --data_path: Path to ERA5 data directory (default: ../data/)

Examples:
    python create_interpolated_dataset.py example_interpolated
    python create_interpolated_dataset.py example_interpolated --no-test
"""

import sys
import os
import argparse
from pathlib import Path

from ERA5.scripts.utilities_era5_old import (
    setup_device,
    load_config,
    split_train_test,
    set_seed,
    prepare_dataframe,
    create_interpolated_sample,
)
from dataset_utilities import GraphLMDBWriter


def validate_interpolated_config(config):
    """Validate interpolated dataset configuration."""
    required_fields = [
        'lat_min', 'lat_max', 'lon_min', 'lon_max',
        'n_points_lat_out', 'n_points_lon_out',
        'n_points_input_min', 'n_points_input_max',
        'k', 'perplexity', 'samples_per_date', 'dates', 'vars'
    ]
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Required configuration field missing: {field}")
    
    # Validate ranges
    if config['lat_min'] >= config['lat_max']:
        raise ValueError("lat_min must be less than lat_max")
    if config['lon_min'] >= config['lon_max']:
        raise ValueError("lon_min must be less than lon_max")
    if config['n_points_lat_out'] <= 0:
        raise ValueError("n_points_lat_out must be positive")
    if config['n_points_lon_out'] <= 0:
        raise ValueError("n_points_lon_out must be positive")
    if config['n_points_input_max'] < config['n_points_input_min']:
        raise ValueError("n_points_input_max must be >= n_points_input_min")


def create_interpolated_lmdb_dataset(config, dates, output_path, device, time, datafolder_path):
    """Create LMDB dataset with interpolated samples."""
    # Check if dataset already exists
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        print(f"\n{'='*60}")
        print(f"ERROR: Dataset already exists at: {output_path}")
        print(f"Please delete or rename the existing dataset before creating a new one.")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Extract parameters
    lat_min = config['lat_min']
    lat_max = config['lat_max']
    lon_min = config['lon_min']
    lon_max = config['lon_max']
    n_lat_out = config['n_points_lat_out']
    n_lon_out = config['n_points_lon_out']
    n_points_input_min = config['n_points_input_min']
    n_points_input_max = config['n_points_input_max']
    k = config['k']
    perplexity = config['perplexity']
    samples_per_date = config['samples_per_date']
    vars_list = config['vars']
    
    # Initialize writer
    writer = GraphLMDBWriter(output_path)
    
    print(f"Creating interpolated dataset: {output_path}")
    print(f"Processing {len(dates)} dates with {samples_per_date} samples per date")
    print(f"Output grid: {n_lat_out}x{n_lon_out} (regular)")
    print(f"Bounds: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"Input nodes: {n_points_input_min}-{n_points_input_max} (random)")
    print(f"Using device: {device}\n")
    
    try:
        for date_idx, date in enumerate(dates):
            print(f"[{date_idx + 1}/{len(dates)}] {date}: ", end='')
            
            # Load data
            df = prepare_dataframe(date, time, vars_list, datafolder_path)
            filename = f"{date}.nc"
            
            # Create multiple samples
            for sample_idx in range(samples_per_date):
                print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {sample_idx + 1}/{samples_per_date}", end='')
                
                data_sample = create_interpolated_sample(
                    df_full=df,
                    date=date,
                    time=time,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    n_lat_out=n_lat_out,
                    n_lon_out=n_lon_out,
                    n_points_input_min=n_points_input_min,
                    n_points_input_max=n_points_input_max,
                    k=k,
                    perplexity=perplexity,
                    device=device,
                    filename=filename
                )
                
                writer.append(data_sample)
            
            print(f"\r[{date_idx + 1}/{len(dates)}] {date}: {samples_per_date}/{samples_per_date} ✓")
    
    finally:
        writer.close()
    
    print(f"\n✓ Dataset creation complete: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create ERA5 interpolated LMDB datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_name",
        help="Name of config file in dataset_configs_interpolated/ (without .json extension)"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip test dataset creation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to ERA5 data directory (default: ../data/)"
    )
    
    args = parser.parse_args()
    config_name = args.config_name
    create_test = not args.no_test
    
    # Setup paths
    parent_dir = Path(__file__).parent.parent
    dataset_config_path = parent_dir / "dataset_configs_interpolated"
    dataset_output_path = parent_dir / "datasets_interpolated"
    dataset_output_path.mkdir(exist_ok=True)
    
    # Data path
    if args.data_path:
        data_path = Path(args.data_path).resolve()
    else:
        data_path = (parent_dir / "data").resolve()
    
    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)
    
    print(f"Data path: {data_path}")
    
    # Setup device
    device = setup_device()
    print(f"Using device: {device}\n")
    
    # Load and validate configuration
    config_file = dataset_config_path / f"{config_name}.json"
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    config = load_config(str(config_file))
    
    try:
        validate_interpolated_config(config)
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    
    # Set random seed
    seed = config.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Split dates
    all_dates = config['dates']
    train_dates, test_dates = split_train_test(
        all_dates,
        train_ratio=0.8,
        shuffle=True,
        random_seed=seed
    )
    
    print(f"Total dates: {len(all_dates)}")
    print(f"  Training: {len(train_dates)}")
    print(f"  Test: {len(test_dates)}\n")
    
    # Create training dataset
    train_output = dataset_output_path / f"{config_name}_train.lmdb"
    create_interpolated_lmdb_dataset(
        config=config,
        dates=train_dates,
        output_path=str(train_output),
        device=device,
        time="00:00",
        datafolder_path=str(data_path)
    )
    
    # Create test dataset
    if create_test and len(test_dates) > 0:
        print()
        test_output = dataset_output_path / f"{config_name}_test.lmdb"
        create_interpolated_lmdb_dataset(
            config=config,
            dates=test_dates,
            output_path=str(test_output),
            device=device,
            time="00:00",
            datafolder_path=str(data_path)
        )
    
    print("\n" + "="*60)
    print("Dataset creation finished successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
