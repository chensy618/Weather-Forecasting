# %%
"""
Random Crop Dataset Creation Script for ERA5 Data
==================================================

Creates train and test LMDB datasets by randomly cropping different geographic
regions from ERA5 data. Each sample uses a randomly selected bounding box and
number of points, creating diverse training data across different areas and scales.

Key features:
- Random bounding box per sample (both train and test)
- Random number of points N within configured [N_min, N_max] range
- Configurable minimum region size

Usage:
    python create_dataset_random_crop_v2.py <config_name>
    
    config_name: Name of config file in configurations/random_crop/dataset_configs/ 
                 (without .json extension)

Examples:
    python create_dataset_random_crop_v2.py base
"""


import sys
import os
import json
import torch
import pandas as pd
import random

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/ERA5/scripts/")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from dataset_utilities import (
    GraphLMDBWriter,
)

from utilities_era5 import ERA5_Graph_Generator


config_id = sys.argv[1]


def sample_random_bbox(min_lat_span=10, min_lon_span=10):
    """
    Generate a random bounding box.
    
    Returns:
        tuple: (lon_min, lat_min, lon_max, lat_max)
    """
    lat_min = random.uniform(-90, 90 - min_lat_span)
    lat_max = random.uniform(lat_min + min_lat_span, 90)
    lon_min = random.uniform(-180, 180 - min_lon_span)
    lon_max = random.uniform(lon_min + min_lon_span, 180)
    return lon_min, lat_min, lon_max, lat_max


def sample_N(N_min, N_max):
    """
    Sample number of points uniformly within range.
    """
    return random.randint(N_min, N_max)


def main(config_id):
    """
    Generate random crop dataset with varying bounding boxes and point counts.
    """
    
    # Open and load the JSON
    with open("../configurations/random_crop/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
        
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    # Load config parameters
    N_min = config['N_min']
    N_max = config['N_max']
    k = config['k']
    perplexity = config['perplexity']
    sampling_method = config['sampling_method']
    min_lat_span = config.get('min_lat_span', 10)
    min_lon_span = config.get('min_lon_span', 10)
    
    train_dates = pd.date_range(config["train_date_start_end"][0], config["train_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    test_dates = pd.date_range(config["test_date_start_end"][0], config["test_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    Generator = ERA5_Graph_Generator(config["data_folder_path"])
    
    for date in train_dates:
        
        for n in range(config["train_samples_per_date"]):
            N = sample_N(N_min, N_max)
            bbox = sample_random_bbox(min_lat_span, min_lon_span)
            Generator.load_data(bbox=bbox, vars=config['vars'], filename=f"{date}.nc")
            
            print(f'{date}: {n+1}/{config["train_samples_per_date"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for date in test_dates:
        
        for n in range(config["test_samples_per_date"]):
            N = sample_N(N_min, N_max)
            bbox = sample_random_bbox(min_lat_span, min_lon_span)
            Generator.load_data(bbox=bbox, vars=config['vars'], filename=f"{date}.nc")
            
            print(f'{date}: {n+1}/{config["test_samples_per_date"]}', end='\r')
            graph = Generator.sample_graph(N=N, k=k, perplexity=perplexity, sampling_method=sampling_method, gen_feature_mode=config["feature_mode"])
            writer.append(graph)
        print()


if __name__ == "__main__":
    main(config_id)
# %%
