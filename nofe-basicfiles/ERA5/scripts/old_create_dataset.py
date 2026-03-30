"""
Dataset Creation Script for ERA5 Weather Data
==============================================

Creates train and test LMDB datasets with graph-structured samples from ERA5 data.

Usage:
    python create_dataset.py <config_name> [--no-test] [--data_path PATH]
    
    config_name: Name of config file in dataset_configs/ (without .json extension)
    --no-test: Optional flag to skip test dataset creation
    --data_path: Path to ERA5 data directory (default: ../data/)

Examples:
    python create_dataset.py example_config
    # Uses default: ../data/
    
    python create_dataset.py example_config --data_path /path/to/ERA5/data
    # Uses custom path
    
    python create_dataset.py example_config --no-test
    # Creates only training dataset
"""

import sys
import os
import argparse
from pathlib import Path

from ERA5.scripts.utilities_era5_old import (
    setup_device,
    load_config,
    validate_config,
    split_train_test,
    create_lmdb_dataset,
    print_dataset_summary,
    set_seed,
)


def main():
    """Main entry point for dataset creation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create ERA5 LMDB datasets from configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config_name",
        help="Name of config file in dataset_configs/ (without .json extension)"
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
    
    # Setup paths (relative to parent directory since we're in scripts/)
    parent_dir = Path(__file__).parent.parent
    dataset_config_path = parent_dir / "dataset_configs"
    dataset_output_path = parent_dir / "datasets"
    dataset_output_path.mkdir(exist_ok=True)
    
    # Data path: use provided path or default to ../data/
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
        validate_config(config)
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid configuration - {e}")
        sys.exit(1)
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}\n")
    
    # Split dates into train and test
    all_dates = config['dates']
    train_dates, test_dates = split_train_test(
        all_dates,
        train_ratio=0.8,
        shuffle=True,
        random_seed=seed
    )
    
    # Print dataset summary
    print_dataset_summary(config_name, train_dates, test_dates, config)
    
    # Create training dataset
    train_output = dataset_output_path / f"{config_name}_train.lmdb"
    create_lmdb_dataset(
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
        create_lmdb_dataset(
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
