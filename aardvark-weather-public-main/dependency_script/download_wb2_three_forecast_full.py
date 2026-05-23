#!/usr/bin/env python3
"""
Download full WeatherBench2 forecast Zarr stores using Python + gcsfs.

Datasets:
  - hres
  - pangu
  - pangu_operational

Default output:
  /cluster/work/projects/nn8106k/siyan/aardvark/datasets/weatherbench2_forecasts_full
"""


"""
mkdir -p dependency_script/.out dependency_script/.err

nohup python -u dependency_script/download_wb2_three_forecast_full.py \
  --datasets hres pangu pangu_operational \
  --resolution 0p25 \
  --out-root /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts \
  --workers 24 \
  > dependency_script/.out/download_wb2_0p25.out \
  2> dependency_script/.err/download_wb2_0p25.err &

nohup python -u dependency_script/download_wb2_three_forecast_full.py \
  --datasets hres pangu pangu_operational \
  --resolution 240x121 \
  --out-root /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts \
  --workers 24 \
  > dependency_script/.out/download_wb2_240x121.out \
  2> dependency_script/.err/download_wb2_240x121.err &

nohup python -u dependency_script/download_wb2_three_forecast_full.py \
  --datasets hres pangu pangu_operational \
  --resolution 64x32 \
  --out-root /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts \
  --workers 24 \
  > dependency_script/.out/download_wb2_64x32.out \
  2> dependency_script/.err/download_wb2_64x32.err &

nohup python -u dependency_script/download_wb2_three_forecast_full.py \
  --datasets hres \
  --resolution 512x256 \
  --out-root /cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts \
  --workers 24 \
  > dependency_script/.out/download_wb2_hres_512x256.out \
  2> dependency_script/.err/download_wb2_hres_512x256.err &

"""


import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gcsfs
from tqdm import tqdm


WB2_FORECAST_PATHS = {
    "0p25": {
        "hres": [
            "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        ],
        "pangu": [
            "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
        ],
        "pangu_operational": [
            "gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr",
            "gs://weatherbench2/datasets/pangu_hres_init/2021_0012_0p25.zarr",
        ],
    },
    "240x121": {
        "hres": [
            "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
        ],
        "pangu": [
            "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr",
        ],
        "pangu_operational": [
            "gs://weatherbench2/datasets/pangu_hres_init/2020_0012_240x121_equiangular_with_poles_conservative.zarr",
            "gs://weatherbench2/datasets/pangu_hres_init/2021_0012_240x121_equiangular_with_poles_conservative.zarr",
        ],
    },
    "64x32": {
        "hres": [
            "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
        ],
        "pangu": [
            "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
        ],
        "pangu_operational": [
            "gs://weatherbench2/datasets/pangu_hres_init/2020_0012_64x32_equiangular_conservative.zarr",
            "gs://weatherbench2/datasets/pangu_hres_init/2021_0012_64x32_equiangular_conservative.zarr",
        ],
    },
    "512x256": {
        "hres": [
            "gs://weatherbench2/datasets/hres/2016-2022-0012-512x256_equiangular_conservative.zarr",
        ],
    },
}


def strip_gs(path: str) -> str:
    return path[len("gs://"):] if path.startswith("gs://") else path


def zarr_name(gs_path: str) -> str:
    return gs_path.rstrip("/").split("/")[-1]


def copy_one(fs: gcsfs.GCSFileSystem, remote_file: str, remote_prefix: str, local_store: Path) -> str:
    rel = Path(remote_file).relative_to(remote_prefix)
    local_file = local_store / rel

    if local_file.exists() and local_file.stat().st_size > 0:
        return "skipped"

    local_file.parent.mkdir(parents=True, exist_ok=True)

    with fs.open(remote_file, "rb") as src, open(local_file, "wb") as dst:
        dst.write(src.read())

    return "downloaded"


def download_gcs_prefix(
    fs: gcsfs.GCSFileSystem,
    gs_path: str,
    local_root: Path,
    workers: int,
) -> None:
    remote_prefix = strip_gs(gs_path).rstrip("/")
    local_store = local_root / zarr_name(gs_path)

    print("=" * 100, flush=True)
    print(f"Remote: gs://{remote_prefix}", flush=True)
    print(f"Local:  {local_store}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print("=" * 100, flush=True)

    print("Listing remote objects...", flush=True)
    files = fs.find(remote_prefix)
    print(f"Number of objects: {len(files)}", flush=True)

    local_store.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(copy_one, fs, remote_file, remote_prefix, local_store)
            for remote_file in files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=zarr_name(gs_path)):
            result = fut.result()
            if result == "downloaded":
                downloaded += 1
            else:
                skipped += 1

    print(f"Finished: {local_store}", flush=True)
    print(f"Downloaded: {downloaded}, skipped: {skipped}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hres", "pangu", "pangu_operational"],
        choices=["hres", "pangu", "pangu_operational"],
    )

    parser.add_argument(
        "--resolution",
        default="0p25",
        choices=["0p25", "240x121", "64x32", "512x256"],
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/cluster/work/projects/nn8106k/siyan/weatherbench2_forecasts"),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Parallel download workers.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    print("WeatherBench2 full forecast download", flush=True)
    print("-" * 100, flush=True)
    print("Datasets:", args.datasets, flush=True)
    print("Resolution:", args.resolution, flush=True)
    print("Out root:", args.out_root, flush=True)
    print("Workers:", args.workers, flush=True)
    print("-" * 100, flush=True)

    fs = gcsfs.GCSFileSystem(token="anon")

    for dataset in args.datasets:
        if dataset not in WB2_FORECAST_PATHS[args.resolution]:
            print(f"Skip {dataset}: no {args.resolution} path available.", flush=True)
            continue

        dataset_out = args.out_root / dataset / args.resolution
        dataset_out.mkdir(parents=True, exist_ok=True)

        for gs_path in WB2_FORECAST_PATHS[args.resolution][dataset]:
            if args.dry_run:
                print(f"[dry-run] {gs_path} -> {dataset_out / zarr_name(gs_path)}", flush=True)
            else:
                download_gcs_prefix(fs, gs_path, dataset_out, workers=args.workers)

    print("\nAll downloads finished.", flush=True)


if __name__ == "__main__":
    main()
