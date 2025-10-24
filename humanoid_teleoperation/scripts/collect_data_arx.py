import asyncio
import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from termcolor import cprint
import threading

# Camera pipeline (reuse original logic)
from multi_realsense import MultiRealSense


def _ensure_root_imports():
    """Allow importing arx_control modules by adding repo root to sys.path.

    This mirrors the path handling in arx_control/examples/joycon_teleop.py.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    return repo_root


class MovingAverage:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = []

    def next(self, val: float) -> float:
        if len(self.window) < self.window_size:
            self.window.append(val)
            return sum(self.window) / max(1, len(self.window))
        else:
            self.window.pop(0)
            self.window.append(val)
            return sum(self.window) / self.window_size

    def get(self) -> float:
        return sum(self.window) / max(1, len(self.window))


class ArxDataCollector:
    """
    Collect synchronized observations from MultiRealSense and ARX5 controller.

    Flow (matching silverscreen_multicam style):
    - Wait loop: keep camera warm, poll Joycon, press X to enter teleop.
    - Teleop loop: per-tick get camera, read Joycon pose, send EEF cmd, save state+cmd, check B to stop.
    - Async lock for camera buffers; frequency pacing via asyncio.sleep.
    """

    def __init__(
        self,
        model: str,
        interface: str,
        fps: int = 50,
        length: int = 2000,
        demo_dir: str = "demo_dir",
        demo_name: str = "demo",
        resize_hw: int = 224,
        save_img: bool = True,
        save_depth: bool = True,
        front_num_points: int = 10000,
    ) -> None:
        self.fps = int(fps)
        self.length = int(length)
        self.demo_dir = demo_dir
        self.demo_name = demo_name
        self.resize_hw = int(resize_hw)
        self.save_img = bool(save_img)
        self.save_depth = bool(save_depth)
        self.front_num_points = int(front_num_points)

        # Async lock for safe writes
        self.lock = asyncio.Lock()

        # Buffers
        self.color_array = []
        self.depth_array = []
        self.cloud_array = []
        self.env_qpos_array = []
        self.action_array = []

        # Camera
        self.camera_context = MultiRealSense(
            use_front_cam=True,
            use_right_cam=False,
            front_num_points=self.front_num_points,
        )

        # Import ARX controller (ensure module path)
        repo_root = _ensure_root_imports()
        arx_dir = os.path.join(repo_root, "arx_control")
        if arx_dir not in sys.path:
            sys.path.append(arx_dir)
        import arx5_interface as arx  # type: ignore
        self._arx = arx
        self.controller = arx.Arx5CartesianController(model, interface)

        # Joycon (no example dependency)
        from arx_control.joycon.joyconrobotics.joyconrobotics import JoyconRobotics  # type: ignore
        robot_config = self.controller.get_robot_config()
        self.joycon = JoyconRobotics(
            device="right",
            translation_frame="local",
            direction_reverse=[1, 1, 1],
            euler_reverse=[-1, -1, 1],
            home_position=self.controller.get_home_pose().tolist()[:3],
            limit_dof=True,
            glimit=[[0.0, -0.5, -0.5, -1.3, -1.3, -1.3], [0.5, 0.5, 0.5, 1.3, 1.3, 1.3]],
            gripper_limit=[0.0, float(robot_config.gripper_width)],
        )

    async def write_data(self, color: Optional[np.ndarray], depth: Optional[np.ndarray], point_cloud: Optional[np.ndarray]):
        async with self.lock:
            if self.save_img and color is not None:
                self.color_array.append(color)
            if self.save_depth and depth is not None:
                self.depth_array.append(depth)
            if point_cloud is not None:
                self.cloud_array.append(point_cloud)

    async def run(self):
        cprint("ARX DataCollector starting...", "cyan")
        measured_duration = MovingAverage(10)
        os.makedirs(self.demo_dir, exist_ok=True)
        record_file_name = os.path.join(self.demo_dir, f"{self.demo_name}.h5")

        try:
            # Warmup
            time.sleep(1.0)
            cprint("Waiting for X to start recording...", "cyan")
            while True:
                _ = self.camera_context()
                x_state = self.joycon.listen_button('x')
                if x_state == 1:
                    cprint("[Joycon] X pressed -> enter teleop loop.", "green")
                    break
                await asyncio.sleep(1.0 / self.fps)

            cprint("Start teleop loop", "green")
            for i in range(self.length):
                start = time.time()

                cam_dict = self.camera_context()
                joy_pose, gripper, _ = self.joycon.get_control()

                # Send EEF command
                eef_cmd = self._arx.EEFState()
                eef_cmd.pose_6d()[:] = np.asarray(joy_pose[:6], dtype=np.float64)
                eef_cmd.gripper_pos = float(gripper)
                eef_cmd.timestamp = self.controller.get_timestamp()
                self.controller.set_eef_cmd(eef_cmd)

                # Resize and buffer camera data
                color_resized = None
                depth_resized = None
                if self.save_img and "color" in cam_dict and cam_dict["color"] is not None:
                    color_resized = cv2.resize(cam_dict["color"], (self.resize_hw, self.resize_hw), interpolation=cv2.INTER_LINEAR)
                if self.save_depth and "depth" in cam_dict and cam_dict["depth"] is not None:
                    depth_resized = cv2.resize(cam_dict["depth"], (self.resize_hw, self.resize_hw), interpolation=cv2.INTER_LINEAR)
                await self.write_data(color_resized, depth_resized, cam_dict.get("point_cloud"))

                # Record joint state and current cmd
                joint_state = self.controller.get_joint_state()
                joint_cmd = self.controller.get_joint_cmd()
                pos_state = np.copy(joint_state.pos())
                pos_cmd = np.copy(joint_cmd.pos())
                env_qpos = np.concatenate([pos_state, np.array([float(joint_state.gripper_pos)], dtype=np.float64)])
                action = np.concatenate([pos_cmd, np.array([float(joint_cmd.gripper_pos)], dtype=np.float64)])
                self.env_qpos_array.append(env_qpos)
                self.action_array.append(action)

                # Stop on B
                b_state = self.joycon.listen_button('b')
                if b_state == 1:
                    cprint("[Joycon] B pressed -> stop recording & exit teleop.", "yellow")
                    break

                duration = time.time() - start
                measured_duration.next(duration)
                fps_now = 1.0 / max(1e-6, duration)
                text = f"time (ms): {measured_duration.get() * 1000.0: >#8.3f} | step: {i} / {self.length} | fps: {fps_now:.3f}"
                print(text, end="\r")
                await asyncio.sleep(1.0 / self.fps)

        finally:
            # Cleanup and save
            self.camera_context.finalize()
            try:
                self.joycon.disconnect()
            except Exception:
                pass
            try:
                self.controller.reset_to_home()
                self.controller.set_to_damping()
            except Exception:
                pass

            # Save H5
            discard_end_length = 0
            seq_length = len(self.action_array)
            color_np = np.array(self.color_array[:seq_length]) if self.save_img else None
            depth_np = np.array(self.depth_array[:seq_length]) if self.save_depth else None
            cloud_np = np.array(self.cloud_array[:seq_length]) if len(self.cloud_array) > 0 else None
            env_qpos_np = np.array(self.env_qpos_array[:seq_length])
            action_np = np.array(self.action_array[:seq_length])

            try:
                import h5py  # type: ignore
            except ImportError:
                raise RuntimeError("h5py is required to save the dataset. Please install it (e.g., pip install h5py).")

            with h5py.File(record_file_name, "w") as f:
                if color_np is not None:
                    f.create_dataset("color", data=color_np[:-discard_end_length])
                if depth_np is not None:
                    f.create_dataset("depth", data=depth_np[:-discard_end_length])
                if cloud_np is not None:
                    f.create_dataset("cloud", data=cloud_np[:-discard_end_length])
                f.create_dataset("env_qpos_proprioception", data=env_qpos_np[:-discard_end_length])
                f.create_dataset("action", data=action_np[:-discard_end_length])

            if color_np is not None:
                cprint(f"color shape: {color_np.shape}", "yellow")
            if depth_np is not None:
                cprint(f"depth shape: {depth_np.shape}", "yellow")
            if cloud_np is not None:
                cprint(f"cloud shape: {cloud_np.shape}", "yellow")
            cprint(f"action shape: {action_np.shape}", "yellow")
            cprint(f"env_qpos shape: {env_qpos_np.shape}", "yellow")
            cprint(f"save data at step: {seq_length} in {record_file_name}", "yellow")


def build_argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Collect data with ARX controller + Realsense")
    parser.add_argument("model", type=str, help="ARX arm model, e.g., X5 or L5")
    parser.add_argument("interface", type=str, help="CAN interface name, e.g., can0")
    parser.add_argument("--fps", type=int, default=50, help="capture frequency (Hz)")
    parser.add_argument("--length", type=int, default=2000, help="number of steps to record")
    parser.add_argument("--demo_dir", type=str, default="demo_dir", help="directory to save H5")
    parser.add_argument("--demo_name", type=str, default="demo", help="file name without extension")
    parser.add_argument("--resize", type=int, default=224, help="color/depth resize square size")
    parser.add_argument("--save_img", type=int, default=1, help="save color images (1/0)")
    parser.add_argument("--save_depth", type=int, default=1, help="save depth maps (1/0)")
    parser.add_argument("--front_points", type=int, default=10000, help="front camera point cloud size")
    return parser


async def main_async(args=None):
    if args is None:
        parser = build_argparser()
        args = parser.parse_args()

    collector = ArxDataCollector(
        model=args.model,
        interface=args.interface,
        fps=args.fps,
        length=args.length,
        demo_dir=args.demo_dir,
        demo_name=args.demo_name,
        resize_hw=args.resize,
        save_img=bool(args.save_img),
        save_depth=bool(args.save_depth),
        front_num_points=args.front_points,
    )

    await collector.run()


def main():
    import argparse
    parser = build_argparser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
