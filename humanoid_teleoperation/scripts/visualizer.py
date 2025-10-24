"""Simple non-blocking point cloud visualizer using Open3D.

Design:
- Keep a single persistent Open3D Visualizer window.
- Each call to `visualize_pointcloud(points)` updates the geometry and returns immediately.
- RGB expected in [0, 255].
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import atexit
from typing import Any, Optional, cast

# Global singletons for persistent window and geometry
_viz: Optional[Any] = None
_pcd: o3d.geometry.PointCloud | None = None
_geom_added: bool = False


def _ensure_visualizer(width: int = 960, height: int = 720, title: str = "PointCloud") -> None:
    global _viz
    if _viz is not None:
        return
    # Create Visualizer via getattr to avoid static attribute issues
    vis_mod = getattr(o3d, 'visualization', None)
    if vis_mod is None:
        raise RuntimeError('open3d.visualization module not available')
    VizCls = getattr(vis_mod, 'Visualizer')
    _viz = VizCls()
    viz = cast(Any, _viz)
    viz.create_window(window_name=title, width=width, height=height)

def close_visualizer() -> None:
    """Close the persistent Open3D visualizer window if open."""
    global _viz, _pcd, _geom_added
    if _viz is not None:
        try:
            _viz.destroy_window()
        except Exception:
            pass
    _viz = None
    _pcd = None
    _geom_added = False


# Ensure cleanup at process exit
atexit.register(close_visualizer)


def visualize_pointcloud(points: np.ndarray) -> None:
    """
    Non-blocking visualization of colored point cloud with Open3D.
    - points: (N,6), columns [x,y,z,r,g,b], rgb in [0,255].
    Behavior:
    - Creates a window on first call, then only updates geometry.
    - Returns immediately after a single render cycle (poll + update + draw).
    """

    _ensure_visualizer()

    # Validate and prepare data
    xyz = points[:, :3].astype(np.float64, copy=False)
    rgb = (points[:, 3:6].astype(np.float64, copy=False) / 255.0).clip(0.0, 1.0)

    assert _viz is not None
    viz = cast(Any, _viz)
    global _pcd, _geom_added

    if not _geom_added:
        # Create and add geometry on first valid data
        _pcd = o3d.geometry.PointCloud()
        pcd = cast(Any, _pcd)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        viz.add_geometry(pcd)
        _geom_added = True

        # Set an initial view around the data
        ctr = viz.get_view_control()
        if ctr is not None:
            center = xyz.mean(axis=0)
            ctr.set_lookat(center.tolist())
            # Look towards +Z (depth increasing), common for RGB-D
            ctr.set_front([0.0, 0.0, 1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            # ctr.set_zoom(0.6)

    else:
        # Update existing geometry
        assert _pcd is not None
        pcd = cast(Any, _pcd)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        viz.update_geometry(pcd)

    # Process a single non-blocking render cycle
    viz.poll_events()
    viz.update_renderer()
