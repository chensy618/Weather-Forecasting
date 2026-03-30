#!/usr/bin/env python3
"""
Subgraph Dataset Creation Script for ERA5 Weather Data
============================================================

Creates train and test LMDB datasets with two-subgraph structure (input and output graphs)
from ERA5 data. This is an alternative to create_dataset.py with a different graph topology.

Graph structure:
- Input graph: Random subset of points with k-NN connectivity
- Output graph: All points with k-NN connectivity  
- Cross edges: Connections from output nodes to nearest input nodes

Usage:
    python create_subgraph_dataset.py <config_name> [--no-test] [--data_path PATH]
    
    config_name: Name of config file in dataset_configs_subgraph/ (without .json extension)
    --no-test: Optional flag to skip test dataset creation
    --data_path: Path to ERA5 data directory (default: ../data/)

Examples:
    python create_subgraph_dataset.py subgraph_config
    # Uses default: ../data/
    
    python create_subgraph_dataset.py subgraph_config --data_path /path/to/ERA5/data
    # Uses custom path
"""

import sys
import os
import argparse
from pathlib import Path

from ERA5.scripts.utilities_era5_old import (
    setup_device,
    load_config,
    validate_subgraph_config,
    split_train_test,
    create_subgraph_lmdb_dataset,
    set_seed,
)


def print_subgraph_summary(config_name, train_dates, test_dates, config):
    """Print summary of subgraph dataset configuration."""
    total_dates = len(train_dates) + len(test_dates)
    samples_per_date = config['samples_per_date']
    
    print("=" * 60)
    print(f"Subgraph Dataset Configuration: {config_name}")
    print("=" * 60)
    print(f"Total dates:          {total_dates}")
    print(f"  Training dates:     {len(train_dates)}")
    print(f"  Test dates:         {len(test_dates)}")
    print(f"Samples per date:     {samples_per_date}")
    print(f"  Total train samples: {len(train_dates) * samples_per_date}")
    print(f"  Total test samples:  {len(test_dates) * samples_per_date}")
    print(f"\nGraph parameters:")
    print(f"  k_in (input):       {config['k_in']}")
    print(f"  k_out (output):     {config['k_out']}")
    print(f"  k_cross (cross):    {config['k_cross']}")
    print(f"  Perplexity:         {config['perplexity']}")
    print(f"  Input nodes:        {config['n_points_input_min']}-{config['n_points_input_max']} (random)")
    print(f"\nVariables: {', '.join(config['vars'])}")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")


def main():
    """Main entry point for subgraph dataset creation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create ERA5 subgraph LMDB datasets from configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_name",
        help="Name of config file in dataset_configs_subgraph/ (without .json extension)"
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
        help="Path to ERA5 data directory (default: ../data/ relative to script)"
    )
    
    args = parser.parse_args()
    config_name = args.config_name
    create_test = not args.no_test
    
    # Setup paths
    parent_dir = Path(__file__).parent.parent
    dataset_config_path = parent_dir / "dataset_configs_subgraph"
    dataset_output_path = parent_dir / "datasets_subgraph"
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
    print(f"Using device: {device}")
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load and validate configuration
    config_file = dataset_config_path / f"{config_name}.json"
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    config = load_config(str(config_file))
    
    try:
        validate_subgraph_config(config)
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
    
    
    
    # --------- Print summary --------------
    print_subgraph_summary(config_name, train_dates, test_dates, config)
    
    # Create training dataset
    train_output = dataset_output_path / f"{config_name}_train.lmdb"
    create_subgraph_lmdb_dataset(
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
        create_subgraph_lmdb_dataset(
            config=config,
            dates=test_dates,
            output_path=str(test_output),
            device=device,
            time="00:00",
            datafolder_path=str(data_path)
        )
    
    print("\n" + "="*60)
    print("Subgraph dataset creation finished successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
