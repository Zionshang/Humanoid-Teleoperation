#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np
import zarr

from open3d_viz import AsyncPointCloudViewer


def _resolve_point_cloud_array(zarr_path: Path):
    root = zarr.open(str(zarr_path), mode="r")
    data_group = root.get("data", None)
    if data_group is None:
        data_group = root

    for key in ("point_cloud", "cloud", "clouds"):
        if key in data_group:
            return data_group[key]

    raise KeyError(f"Could not find a point cloud dataset under {zarr_path}")


def visualize_point_clouds(
    zarr_path: Path,
    start: int,
    end: int,
    stride: int,
    delay: float,
    loop: bool,
    point_size: float,
    window_name: str,
):
    pc_array = _resolve_point_cloud_array(zarr_path)
    total_frames = pc_array.shape[0]

    if end < 0 or end > total_frames:
        end = total_frames
    start = max(0, min(start, total_frames - 1))
    stride = max(1, stride)

    viewer = AsyncPointCloudViewer(point_size=point_size, window_name=window_name)

    print(
        f"Visualizing point clouds from {zarr_path} "
        f"[frames {start}:{end}:{stride}, total {total_frames}]"
    )

    try:
        while True:
            for idx in range(start, end, stride):
                pc = np.asarray(pc_array[idx])
                viewer.update(pc)
                print(f"Frame {idx}/{total_frames}", end="\r", flush=True)
                time.sleep(delay)
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print("\nViewer closed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize point clouds stored inside a converted Zarr dataset."
    )
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr dataset directory.")
    parser.add_argument("--start", type=int, default=0, help="Starting frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive).")
    parser.add_argument("--stride", type=int, default=2, help="Frame sampling stride.")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between frames (seconds).")
    parser.add_argument("--loop", type=int, default=1, help="Loop through the sequence indefinitely.")
    parser.add_argument("--point-size", type=float, default=2.0, dest="point_size")
    parser.add_argument("--window-name", type=str, default="Zarr Point Clouds", dest="window_name")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize_point_clouds(
        zarr_path=Path(args.zarr_path),
        start=args.start,
        end=args.end,
        stride=args.stride,
        delay=args.delay,
        loop=bool(args.loop),
        point_size=args.point_size,
        window_name=args.window_name,
    )


if __name__ == "__main__":
    main()
