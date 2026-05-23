#!/bin/bash

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

export AARDVARK_ROOT=/cluster/work/projects/nn8106k/siyan/aardvark
export AARDVARK_REPO=$HOME/github/Weather-Forecasting/aardvark-weather-public-main

source $AARDVARK_ROOT/envs/aardvark/bin/activate

export AARDVARK_DATA_DIR=$AARDVARK_ROOT/datasets
export AARDVARK_RAW_DIR=$AARDVARK_ROOT/datasets/raw
export AARDVARK_PROCESSED_DIR=$AARDVARK_ROOT/datasets/processed
export AARDVARK_CKPT_DIR=$AARDVARK_ROOT/checkpoints

export PIP_CACHE_DIR=$AARDVARK_ROOT/cache/pip
export TORCH_HOME=$AARDVARK_ROOT/cache/torch
export HF_HOME=$AARDVARK_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$AARDVARK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$AARDVARK_ROOT/cache/xdg

export PYTHONPATH=$AARDVARK_REPO:$PYTHONPATH

cd $AARDVARK_REPO

echo "Aardvark environment loaded."
echo "Python: $(which python)"
echo "Data: $AARDVARK_DATA_DIR"
echo "Checkpoints: $AARDVARK_CKPT_DIR"
