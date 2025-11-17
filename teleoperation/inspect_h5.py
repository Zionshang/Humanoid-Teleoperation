#!/usr/bin/env python3
"""
Quickly inspect a .h5 demonstration file by printing dataset metadata such as
shape, dtype, and optionally min/max values.
"""
import argparse
import os
from pathlib import Path

import h5py


def summarize_dataset(name: str, dataset: h5py.Dataset, show_stats: bool) -> str:
    info = f"{name}: shape={dataset.shape}, dtype={dataset.dtype}"
    if show_stats and dataset.size > 0:
        try:
            data = dataset[...]
            info += f", min={data.min():.3f}, max={data.max():.3f}"
        except Exception as exc:  # pragma: no cover - stats are best effort
            info += f", stats_error={exc}"
    return info


def walk_h5(file_path: Path, show_stats: bool) -> None:
    print(f"\n=== {file_path} ===")
    if not file_path.exists():
        print("File not found.")
        return

    with h5py.File(file_path, "r") as h5_file:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print("  " + summarize_dataset(name, obj, show_stats))

        h5_file.visititems(visitor)


def parse_args():
    parser = argparse.ArgumentParser(description="Print basic information about .h5 files.")
    parser.add_argument("files", nargs="+", help="One or more .h5 file paths or directories.")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute min/max values for each dataset (may be slow for large files).",
    )
    return parser.parse_args()


def collect_h5_paths(inputs):
    paths = []
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix == ".h5":
            paths.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*.h5")):
                paths.append(child)
        else:
            print(f"Skipping {path}: not an .h5 file or directory.")
    return paths


def main():
    args = parse_args()
    h5_paths = collect_h5_paths(args.files)
    if not h5_paths:
        print("No .h5 files found.")
        return
    for file_path in h5_paths:
        walk_h5(file_path, args.stats)


if __name__ == "__main__":
    main()
