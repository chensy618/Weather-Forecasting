import sys
import os
import json
import torch
import pandas as pd


# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.chdir("/Volumes/T7-storage/NOFE/ERA5/scripts/" )
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from dataset_utilities import (
    GraphLMDBWriter,
    #GraphLMDBReader,
)

from utilities_era5 import (
    ERA5_Graph_Generator,
    track_time,
    _function_timings
)

config_id = sys.argv[1]



#%%

def main(config_id): #regular method
    """
    :param config: Description
    """
    
    # Open and load the JSON
    with open("../configurations/subgraph/dataset_configs/"+config_id+".json", "r") as f:
        config = json.load(f)
    
    print("Configuration:\n")
    print(*[f"{k}: {v}" for k, v in config.items()], sep="\n")
    
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_train.lmdb'):
        print('dataset already exists -- STOP')
        return()
    if os.path.exists(f'{config["dataset_save_path"]}/{config_id}_test.lmdb'):
        print('dataset already exists -- STOP')
        return()
    
    bbox = config['lon_min'], config['lat_min'], config['lon_max'], config['lat_max']
    train_dates= pd.date_range(config["train_date_start_end"][0], config["train_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    test_dates= pd.date_range(config["test_date_start_end"][0], config["test_date_start_end"][1]).strftime("%Y-%m-%d").tolist()
    
    # Train Set
    print('\nGenerate Training Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_train.lmdb')
    data_folder = config["data_folder_path"]
    Generator = ERA5_Graph_Generator(data_folder) 

    for date in train_dates:
        Generator.load_data(bbox=bbox, vars=config['vars'], filename=f'{date}.nc')
        
        # create once beofre loop if output gaph is regular -> more efficient
        if config['output_sampling_method'] == "regular_subset":
            out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        for n in range(config["train_samples_per_date"]):
            
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            print(f'{date}: {n+1}/{config["train_samples_per_date"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()
    
    # Test Set
    print('\nGenerate Test Set\n')
    
    writer = GraphLMDBWriter(f'{config["dataset_save_path"]}/{config_id}_test.lmdb')
    
    for date in test_dates:
        Generator.load_data(bbox=bbox, vars=config['vars'], filename=f'{date}.nc')
        
        # create once beofre loop if output gaph is regular -> more efficient
        if config['output_sampling_method'] == "regular_subset":
            out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
        for n in range(config["test_samples_per_date"]):
            
            # if output graph is not reused, it has to be created inside the loop
            if config['output_sampling_method'] != "regular_subset":
                out_graph = Generator.sample_graph(N=config["N_out"], k=config["k_out"], perplexity=config["perplexity"], sampling_method=config["output_sampling_method"], gen_feature_mode=config["feature_mode"])
            
            print(f'{date}: {n+1}/{config["test_samples_per_date"]}', end='\r')
            in_graph = Generator.sample_graph(N=config["N_in"], k=config["k_in"], perplexity=config["perplexity"], sampling_method=config["input_sampling_method"], gen_feature_mode=config["feature_mode"])
            cross_graph = Generator.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=config["k_cross"])
            cross_graph.cross["x"] = Generator.construct_output_node_features(cross_graph=cross_graph, interpolate=config["interpolate_graph_features"], gen_feature_mode=config["feature_mode"])
            writer.append(cross_graph)
        print()

    # --- Print execution times --- #
    print("\nTiming summary:")

    # dynamic column widths
    name_width = max(len(name) for name in _function_timings) + 2
    calls_width = max(len(str(stats["calls"])) for stats in _function_timings.values())
    total_width = max(len(f"{stats['total']:.4f}") for stats in _function_timings.values())
    avg_width = max(
        len(f"{stats['total'] / stats['calls']:.4f}")
        for stats in _function_timings.values()
    )

    # headers
    header = (
        f"{'Function':<{name_width}} | "
        f"{'Calls':>{calls_width}} | "
        f"{'Total (s)':>{total_width}} | "
        f"{'Avg (s)':>{avg_width}}"
    )
    print(header)
    print("-" * len(header))

    # rows
    for name, stats in sorted(
        _function_timings.items(),
        key=lambda x: x[1]["total"],
        reverse=True,
    ):
        total = stats["total"]
        calls = stats["calls"]
        avg = total / calls

        print(
            f"{name:<{name_width}} | "
            f"{calls:>{calls_width}d} | "
            f"{total:>{total_width}.4f} | "
            f"{avg:>{avg_width}.4f}"
        )




if __name__ == "__main__":
    main(config_id)



