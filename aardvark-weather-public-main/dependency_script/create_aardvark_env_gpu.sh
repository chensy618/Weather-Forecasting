#!/bin/bash
#SBATCH --job-name=create_aardvark_env
#SBATCH --account=nn8106k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=dependency_script/.out/create_aardvark_env_%j.out
#SBATCH --error=dependency_script/.err/create_aardvark_env_%j.err

set -e

echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"

module --force purge
module load NRIS/GPU
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/13.0.0

export AARDVARK_ROOT=/cluster/work/projects/nn8106k/siyan/aardvark
export AARDVARK_REPO=$HOME/github/Weather-Forecasting/aardvark-weather-public-main

mkdir -p $AARDVARK_ROOT/envs
mkdir -p $AARDVARK_ROOT/cache/pip
mkdir -p $AARDVARK_ROOT/cache/torch
mkdir -p $AARDVARK_ROOT/cache/huggingface
mkdir -p $AARDVARK_ROOT/cache/xdg
mkdir -p $AARDVARK_ROOT/datasets/raw
mkdir -p $AARDVARK_ROOT/datasets/processed
mkdir -p $AARDVARK_ROOT/checkpoints

export PIP_CACHE_DIR=$AARDVARK_ROOT/cache/pip
export TORCH_HOME=$AARDVARK_ROOT/cache/torch
export HF_HOME=$AARDVARK_ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$AARDVARK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$AARDVARK_ROOT/cache/xdg

echo "System Python:"
which python
python --version

echo "Remove old environment..."
rm -rf $AARDVARK_ROOT/envs/aardvark

echo "Create venv..."
python -m venv --copies $AARDVARK_ROOT/envs/aardvark

echo "Activate venv..."
source $AARDVARK_ROOT/envs/aardvark/bin/activate

echo "Venv Python:"
which python
python -c "import sys; print(sys.executable)"

echo "Upgrade pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Install PyTorch..."
python -m pip install torch torchvision torchaudio

echo "Install Aardvark dependencies..."
python -m pip install \
  numpy pandas scipy xarray matplotlib networkx tqdm pyyaml requests pillow \
  zarr netCDF4 h5py cftime fsspec gcsfs \
  huggingface-hub safetensors timm plotly wandb \
  aiohttp appdirs asciitree black cachetools click \
  eumdac fasteners frozenlist geographiclib gitpython \
  google-cloud-storage numcodecs ratelimiter scienceplots tenacity tokenize-rt \
  cfgrib eccodes shapely pyproj geopy

export PYTHONPATH=$AARDVARK_REPO:$PYTHONPATH
cd $AARDVARK_REPO

echo "Test torch..."
python - << 'PYEOF'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYEOF

echo "Test aardvark import..."
python - << 'PYEOF'
import aardvark
print("Aardvark import ok")
PYEOF

echo "Aardvark environment created successfully at:"
echo "$AARDVARK_ROOT/envs/aardvark"
echo "Finished at $(date)"