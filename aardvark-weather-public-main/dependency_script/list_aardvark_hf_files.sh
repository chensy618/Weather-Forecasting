#!/bin/bash
#SBATCH --job-name=list_aardvark_files
#SBATCH --account=nn8106k
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=dependency_script/.out/list_aardvark_files_%j.out
#SBATCH --error=dependency_script/.err/list_aardvark_files_%j.err

set -e

module --force purge
module load NRIS/Login
module load Python/3.11.5-GCCcore-13.2.0

export AARDVARK_ROOT=/cluster/work/projects/nn8106k/siyan/aardvark
source $AARDVARK_ROOT/envs/aardvark_download/bin/activate

export HF_HOME=$AARDVARK_ROOT/cache/huggingface

python - << 'PYEOF'
from huggingface_hub import HfApi

repo_id = "av555/aardvark-weather"
api = HfApi()
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

print("Total files:", len(files))

print("\n=== trained_model ===")
for f in files:
    if f.startswith("trained_model"):
        print(f)

print("\n=== sample_data ===")
for f in files:
    if f.startswith("sample_data"):
        print(f)

print("\n=== files containing 2016 or 2018 ===")
for f in files:
    if "2016" in f or "2018" in f:
        print(f)

print("\n=== training_data preview ===")
count = 0
for f in files:
    if f.startswith("training_data"):
        print(f)
        count += 1
        if count >= 200:
            break
PYEOF
