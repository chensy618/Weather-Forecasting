#!/usr/bin/env python3
"""
Training Script for Subgraph Neural Operator Models
===================================================

Trains a subgraph-based graph neural network model on weather/satellite data
using separate train and test LMDB datasets with input/output graph structure.

Usage:
    python train_subgraph.py <data_source>/<training_config_name>

Examples:
    python train_subgraph.py ERA5/example_subgraph_training
    python train_subgraph.py alphaEarth/example_subgraph_training
"""

import sys
import json
import os
import time
import random
import argparse
from pathlib import Path
from time import localtime, strftime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.utilities import (
    KernelNN_subgraph,
    GraphLMDBReader,
    setup_device,
    custom_KL_loss,
    get_output_affinities,
)


# Async data loading setup
executor = ThreadPoolExecutor(max_workers=1)

def load_batch(reader, index):
    """Helper function for async batch loading."""
    return reader[index]


def get_output_affinities_subgraph(output, edge_idx):
    """
    Compute output affinities from subgraph model output.
    
    For subgraph models, output is from the output nodes only.
    Uses output edges (within output graph) for affinity computation.
    
    Args:
        output: Model output embeddings [N_out, D]
        edge_idx: Output graph edge index [2, E_out]
        
    Returns:
        Normalized affinities [N_out, k_out]
    """
    return get_output_affinities(output, edge_idx)


def main():
    """Main training loop for subgraph model."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train subgraph neural operator models on weather/satellite data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python train_subgraph.py ERA5/example_subgraph_training\n'
               '  python train_subgraph.py alphaEarth/example_subgraph_training --wandb\n'
    )
    parser.add_argument('config_path', metavar='<data_source>/<training_config_name>',
                        help='Path to training configuration (e.g., ERA5/example_subgraph_training)')
    parser.add_argument('--wandb', '-wb', action='store_true',
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    config_path_arg = args.config_path
    use_wandb = args.wandb
    
    # Parse data source and config name
    if '/' in config_path_arg:
        data_source, training_config_id = config_path_arg.split('/', 1)
    else:
        print("Warning: Please specify data source (e.g., ERA5/config_name)")
        print("Assuming ERA5 for backward compatibility...\n")
        data_source = "ERA5"
        training_config_id = config_path_arg
    
    # Setup paths (relative to script directory)
    base_dir = Path(__file__).parent
    data_dir = base_dir / data_source
    
    if not data_dir.exists():
        print(f"Error: Data source directory not found: {data_dir}")
        print(f"Available directories: {[d.name for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]}")
        sys.exit(1)
    
    training_config_path = data_dir / "configurations" / "subgraph" / "training_configs"
    model_config_path = data_dir / "configurations" / "subgraph" / "model_configs"
    dataset_configs_path = data_dir / "configurations" / "subgraph" / "dataset_configs"
    dataset_path = data_dir / "datasets/subgraph"
    
    # Output paths - project-specific directories
    trained_models_path = data_dir / "trained_models/subgraph"
    training_logs_path = data_dir / "trained_models/subgraph/logs"

    
    # Create output directories
    trained_models_path.mkdir(parents=True, exist_ok=True)
    (training_logs_path / "loss_records").mkdir(parents=True, exist_ok=True)
    (training_logs_path / "configs").mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device()
    print(f"Using device: {device}\n")
    
    # Load configurations
    with open(training_config_path / f"{training_config_id}.json", "r") as f:
        train_dict = json.load(f)
    
    dataset_id = train_dict['dataset_id']
    model_config_id = train_dict['model_id']
    
    with open(model_config_path / f"{model_config_id}.json", "r") as f:
        model_dict = json.load(f)
    
    with open(dataset_configs_path / f"{dataset_id}.json", "r") as f:
        dataset_dict = json.load(f)
    
    # Create model ID with timestamp
    time_string = strftime("%Y-%m-%d_t%H-%M-%S", localtime())
    model_id = f"{model_config_id}-subgraph-{time_string}"
    
    # Merge all configs
    network_dict = {**train_dict, **model_dict, **dataset_dict}
    
    # Set random seed for reproducibility
    seed = train_dict.get('random_seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Extract training parameters
    learning_rate = train_dict['learning_rate']
    scheduler_step = train_dict['scheduler_step']
    scheduler_gamma = train_dict['scheduler_gamma']
    epochs = train_dict['epochs']
    batch_size = train_dict['batch_size']
    
    print(f"Data source: {data_source}")
    print(f"Configuration: {training_config_id}")
    print(f"Model ID: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {data_source}/trained_models/")
    print(f"Weights & Biases: {'enabled' if use_wandb else 'disabled'}\n")
    
    # Initialize Weights & Biases (optional)
    if use_wandb:
        import wandb
        wandb.init(
            project=f"NOFE-{data_source.lower()}",
            config=network_dict,
            name=model_id,
            settings=wandb.Settings(code_dir=".")
        )
    
    # Load datasets (separate train and test)
    train_dataset_path = dataset_path / f"{dataset_id}_train.lmdb"
    test_dataset_path = dataset_path / f"{dataset_id}_test.lmdb"
    
    if not train_dataset_path.exists():
        print(f"Error: Training dataset not found: {train_dataset_path}")
        sys.exit(1)
    
    if not test_dataset_path.exists():
        print(f"Warning: Test dataset not found: {test_dataset_path}")
        print("Continuing with train dataset only...")
        test_reader = None
    else:
        test_reader = GraphLMDBReader(str(test_dataset_path))
        print(f"Test dataset: {len(test_reader)} samples")
    
    train_reader = GraphLMDBReader(str(train_dataset_path))
    print(f"Train dataset: {len(train_reader)} samples")
    
    # Use all training data (no manual split)
    train_indices = list(range(len(train_reader)))
    
    # Use test dataset as validation set
    if test_reader is not None:
        val_indices = list(range(len(test_reader)))
        val_reader = test_reader
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples (from test set): {len(val_indices)}\n")
    else:
        # Fallback: if no test dataset, split train data
        print("Warning: No test dataset found, falling back to train/val split (90/10)")
        train_count = len(train_reader)
        train_size = int(0.9 * train_count)
        indices = list(range(train_count))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        val_reader = train_reader
        print(f"Train samples: {len(train_indices)}")
        print(f"Val samples: {len(val_indices)}\n")
    
    # Create model
    # Support both old (single node_features) and new (separate input/output) configs
    # For backward compatibility, fall back to 'node_features' if separate dimensions not specified
    input_node_dim = model_dict.get('input_node_features', model_dict.get('node_features', 1))
    output_node_dim = model_dict.get('output_node_features', model_dict.get('node_features', 1))
    
    model = KernelNN_subgraph(
        width=model_dict['width'],
        ker_width=model_dict['ker_width'],
        depth=model_dict['depth'],
        ker_in=model_dict['edge_attr'],
        known_node_dim=input_node_dim,  # input feature dimension
        unknown_node_dim=output_node_dim,  # output feature dimension
        out_width=model_dict['out_width'],
    ).to(device)
    
    print(f"Model: KernelNN_subgraph")
    print(f"  Width: {model_dict['width']}")
    print(f"  Depth: {model_dict['depth']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    
    # Training records
    loss_record = []
    val_loss_record = []
    epoch_record = []
    time_record = []
    lr_record = []
    
    # Training loop
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Epoch 0: Initial validation (before training)
    print("Epoch 0: Evaluating initial model...")
    model.eval()
    initial_val_losses = []
    
    for j, val_ind in enumerate(val_indices):
        print(f'Epoch 0/{epochs} - Validation: {j+1}/{len(val_indices)}', end='\r')
        batch = load_batch(val_reader, val_ind)
        batch = batch.to(device)
        
        with torch.no_grad():
            # Forward pass for subgraph model
            out = model(batch)
            # Compute affinities using output graph edges
            q = get_output_affinities_subgraph(out, batch.target.edge_index)
            loss = custom_KL_loss(batch.target.input_affinities, q).mean()
            initial_val_losses.append(loss.item())
    
    initial_val_loss = np.mean(initial_val_losses)
    loss_record.append(np.nan)  # No training loss for epoch 0
    val_loss_record.append(initial_val_loss)
    epoch_record.append(0)
    lr_record.append(learning_rate)
    time_record.append("00:00")
    
    if use_wandb:
        wandb.log({
            'epoch': 0,
            'ep_val_loss': initial_val_loss,
        })
    
    #print(f'\nEpoch: 0/{epochs} - Val Loss: {round(initial_val_loss, 4)} (before training)\n')
    print(f'\nVal Loss: {initial_val_loss} (before training)\n')
    
    future = None
    
    for ep in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        all_losses = []
        
        # Training
        for i, train_ind in enumerate(train_indices):
            print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(train_indices)}", end="", flush=True)
            
            # Prefetch next batch
            if i + 1 < len(train_indices):
                future = executor.submit(load_batch, train_reader, train_indices[i + 1])
            
            # Load current batch
            if i == 0:
                batch = load_batch(train_reader, train_ind)
            else:
                batch = future.result()
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass for subgraph model
            out = model(batch)
            # Compute affinities using output graph edges
            q = get_output_affinities_subgraph(out, batch.target.edge_index)
            loss = custom_KL_loss(batch.target.input_affinities, q).mean()
            all_losses.append(loss.item())
            
            if use_wandb:
                wandb.log({'loss': loss.item()})
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        avg_loss = np.mean(all_losses)
        loss_record.append(avg_loss)
        epoch_record.append(ep)
        lr_record.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        future_val = None
        
        for j, val_ind in enumerate(val_indices):
            print(f"\r\033[KEpoch {ep}/{epochs} - Training: {i+1}/{len(train_indices)} - Validation: {j+1}/{len(val_indices)}", end="", flush=True)
            # Prefetch next batch
            if j + 1 < len(val_indices):
                future_val = executor.submit(load_batch, val_reader, val_indices[j + 1])
            
            # Load current batch
            if j == 0:
                batch = load_batch(val_reader, val_ind)
            else:
                batch = future_val.result()
            
            batch = batch.to(device)
            
            with torch.no_grad():
                # Forward pass for subgraph model
                out = model(batch)
                # Compute affinities using output graph edges
                q = get_output_affinities_subgraph(out, batch.target.edge_index)
                loss = custom_KL_loss(batch.target.input_affinities, q).mean()
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        val_loss_record.append(val_loss)
        
        # Epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        minutes, seconds = divmod(int(epoch_duration), 60)
        time_str = f"{minutes:02}:{seconds:02}"
        time_record.append(time_str)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': ep,
                'ep_loss': avg_loss,
                'ep_val_loss': val_loss,
            })
        
        #print(f'Epoch: {ep}/{epochs} - Loss: {round(avg_loss, 4)}\t|  Val Loss: {round(val_loss, 4)}\t|  LR: {round(scheduler.get_last_lr()[0], 7)}\t|  Time: {time_str}')
        print(f'\nLoss: {avg_loss}\t|  Val Loss: {val_loss}\t|  LR: {round(scheduler.get_last_lr()[0], 7)}\t|  Time: {time_str}')
        
        # Save checkpoint every 5 epochs
        if ep % 5 == 0:
            loss_df = pd.DataFrame({
                'epoch': epoch_record,
                'loss': loss_record,
                'val_loss': val_loss_record,
                'lr': lr_record,
                'time': time_record
            })
            
            torch.save(model.state_dict(), trained_models_path / f"{model_id}.pth")
            loss_df.to_csv(training_logs_path / "loss_records" / f"{model_id}.csv")
            
            with open(training_logs_path / "configs" / f"{model_id}.json", 'w') as f:
                json.dump(network_dict, f, indent=4)
            
            print(f'✓ Checkpoint saved: {model_id}\n')
    
    # Final save
    loss_df = pd.DataFrame({
        'epoch': epoch_record,
        'loss': loss_record,
        'val_loss': val_loss_record,
        'lr': lr_record,
        'time': time_record
    })
    
    torch.save(model.state_dict(), trained_models_path / f"{model_id}.pth")
    loss_df.to_csv(training_logs_path / "loss_records" / f"{model_id}.csv")
    
    with open(training_logs_path / "configs" / f"{model_id}.json", 'w') as f:
        json.dump(network_dict, f, indent=4)
    
    print("\n" + "="*60)
    print(f'✓ Training complete!')
    print(f'Model saved: {model_id}')
    print("="*60)
    
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
