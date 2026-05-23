#!/usr/bin/env python3
"""
Download all official Aardvark assets from Hugging Face.

This downloads:
1. trained_model/**  -> checkpoints/
2. sample_data/**    -> datasets/
3. training_data/**  -> datasets/

Recommended to run on Olivia login node inside aardvark_download environment:

    cd /cluster/work/projects/nn8106k/siyan/aardvark
    source envs/aardvark_download/bin/activate

    cd ~/github/Weather-Forecasting/aardvark-weather-public-main
    python dependency_script/download_aardvark_all_assets.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "av555/aardvark-weather"
REPO_TYPE = "dataset"


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total


def list_remote_files(repo_id: str, repo_type: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    return sorted(files)


def print_remote_summary(files: Iterable[str]) -> None:
    files = list(files)

    groups = {
        "trained_model": [f for f in files if f.startswith("trained_model/")],
        "sample_data": [f for f in files if f.startswith("sample_data/")],
        "training_data": [f for f in files if f.startswith("training_data/")],
    }

    print(f"Total remote files: {len(files)}")
    for name, group_files in groups.items():
        print(f"{name}: {len(group_files)} files")
        for f in group_files[:20]:
            print(f"  {f}")
        if len(group_files) > 20:
            print(f"  ... ({len(group_files) - 20} more)")
        print()


def download_pattern(
    repo_id: str,
    repo_type: str,
    allow_patterns: list[str],
    local_dir: Path,
    cache_dir: Path,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Downloading patterns: {allow_patterns}")
    print(f"Local dir: {local_dir}")
    print(f"Cache dir: {cache_dir}")
    print("=" * 100)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        resume_download=True,
    )


def write_manifest(root: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(f"Aardvark assets root: {root}\n\n")

        for subdir in [
            root / "checkpoints",
            root / "datasets" / "sample_data",
            root / "datasets" / "training_data",
            root / "cache" / "huggingface",
        ]:
            f.write(f"{subdir}: {human_size(dir_size(subdir))}\n")

        f.write("\nFiles preview:\n")
        count = 0
        for p in sorted(root.rglob("*")):
            if p.is_file():
                f.write(str(p) + "\n")
                count += 1
                if count >= 300:
                    f.write("... truncated after 300 files\n")
                    break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all official Aardvark assets from Hugging Face."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/cluster/work/projects/nn8106k/siyan/aardvark"),
        help="Aardvark project root on Olivia.",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help="Hugging Face dataset repo id.",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list remote files and do not download.",
    )

    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Skip trained_model download.",
    )

    parser.add_argument(
        "--skip-sample-data",
        action="store_true",
        help="Skip sample_data download.",
    )

    parser.add_argument(
        "--skip-training-data",
        action="store_true",
        help="Skip training_data download.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    cache_dir = root / "cache" / "huggingface"

    checkpoints_dir = root / "checkpoints"
    datasets_dir = root / "datasets"
    sample_dir = datasets_dir / "sample_data"
    training_dir = datasets_dir / "training_data"
    logs_dir = root / "logs"

    for d in [
        checkpoints_dir,
        sample_dir,
        training_dir,
        cache_dir,
        logs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(root / "cache" / "xdg")

    print("Aardvark download configuration")
    print("-" * 100)
    print(f"Repo: {args.repo_id}")
    print(f"Root: {root}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"Sample data dir: {sample_dir}")
    print(f"Training data dir: {training_dir}")
    print(f"HF cache dir: {cache_dir}")
    print("-" * 100)

    files = list_remote_files(args.repo_id, REPO_TYPE)
    print_remote_summary(files)

    if args.list_only:
        print("List-only mode. No files downloaded.")
        return

    if not args.skip_checkpoints:
        download_pattern(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=["trained_model/**"],
            local_dir=checkpoints_dir,
            cache_dir=cache_dir,
        )

    if not args.skip_sample_data:
        download_pattern(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=["sample_data/**"],
            local_dir=datasets_dir,
            cache_dir=cache_dir,
        )

    if not args.skip_training_data:
        download_pattern(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=["training_data/**"],
            local_dir=datasets_dir,
            cache_dir=cache_dir,
        )

    manifest_path = logs_dir / "download_manifest.txt"
    write_manifest(root, manifest_path)

    print("\nDownload complete.")
    print("-" * 100)
    print(f"Checkpoints: {human_size(dir_size(checkpoints_dir))}")
    print(f"Sample data: {human_size(dir_size(sample_dir))}")
    print(f"Training data: {human_size(dir_size(training_dir))}")
    print(f"Manifest: {manifest_path}")
    print("-" * 100)


if __name__ == "__main__":
    main()
