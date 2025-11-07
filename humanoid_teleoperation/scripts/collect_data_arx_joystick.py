import asyncio
import os
import sys
import time
from datetime import datetime
from collections import deque
from typing import Optional

import cv2
import numpy as np
from termcolor import cprint
import threading
import h5py

# Camera pipeline (reuse original logic)
from multi_realsense import MultiRealSense

# Ensure arx_control package is importable from repo root, and import arx5_interface globally
this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(this_dir, os.pardir, os.pardir))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)
_arx_dir = os.path.join(_repo_root, "arx_control")
if _arx_dir not in sys.path:
    sys.path.append(_arx_dir)
import arx5_interface as arx
from arx_control.joystick import JoystickRobotics, XboxButton

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
        fps: int = 8,
        length: int = 2000,
        demo_dir: str = "demo_dir",
        resize_hw: int = 224,
        save_img: bool = True,
        save_depth: bool = True,
    ) -> None:
        self.fps = int(fps)
        self.length = int(length)
        self.demo_dir = demo_dir
        self.resize_hw = int(resize_hw)
        self.save_img = bool(save_img)
        self.save_depth = bool(save_depth)

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
            front_num_points=10000,
            front_z_far=0.8, front_z_near=0.1,
        )

        # Init ARX controller
        self.controller = arx.Arx5CartesianController(model, interface)
        robot_config = self.controller.get_robot_config()
        cprint("Resetting arm to home before teleop...", "cyan")
        self.controller.reset_to_home()

        # open the gripper
        eef_cmd = self.controller.get_eef_state()
        eef_cmd.gripper_pos = robot_config.gripper_width
        eef_cmd.timestamp = self.controller.get_timestamp() + 1.0
        self.controller.set_eef_cmd(eef_cmd)
        
        # Init Joystick
        self.joystick = JoystickRobotics(
        home_position=self.controller.get_home_pose().tolist()[:3],
        home_gripper=robot_config.gripper_width,
        ee_limit=[[0.0, -0.5, -0.5, -1.8, -1.6, -1.6], [0.7, 0.5, 0.5, 1.8, 1.6, 1.6]],
        gripper_limit=[0.0, robot_config.gripper_width],
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
        os.makedirs(self.demo_dir, exist_ok=True)
     
        # Warmup
        time.sleep(1.0)
        cprint("Waiting for X to start recording...", "cyan")
        while True:
            _ = self.camera_context()
            _, _, control_button = self.joystick.get_control()
            if control_button == XboxButton.X:
                cprint("[Joystick] X pressed -> enter teleop loop.", "green")
                break
            await asyncio.sleep(1.0 / self.fps)

        cprint("Start teleop loop", "green")
        period = 1.0 / float(self.fps)
        for i in range(self.length):
            start = time.time()

            cam_dict = self.camera_context()
            joy_pose, gripper_pose, control_button = self.joystick.get_control()

            # Send EEF command
            eef_cmd = arx.EEFState()
            eef_cmd.pose_6d()[:] = joy_pose
            eef_cmd.gripper_pos = gripper_pose
            eef_cmd.timestamp = self.controller.get_timestamp() + 0.1
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
            if control_button == XboxButton.B:
                cprint("[Joystick] B pressed -> stop recording & exit teleop.", "yellow")
                break

            elapsed = time.time() - start
            wait = max(0.0, period - elapsed)
            fps_now = 1.0 / max(1e-6, elapsed)
            print(f"elapsed time (ms): {elapsed * 1000.0: >#8.3f} | step: {i} / {self.length} | fps: {fps_now:.3f}", end="\r")

            await asyncio.sleep(wait)

    
        # Cleanup and save
        self.camera_context.finalize()
        self.controller.reset_to_home()
        self.controller.set_to_damping()

        # Save H5（以结束时间生成文件名）
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_file_name = os.path.join(self.demo_dir, f"{base_name}.h5")

        discard_end_length = 5
        with h5py.File(record_file_name, "w") as f:
            seq_length = len(self.action_array)
            color_array = np.array(self.color_array)[:seq_length]
            depth_array = np.array(self.depth_array)[:seq_length]
            cloud_array = np.array(self.cloud_array)[:seq_length]
            env_qpos_array = np.array(self.env_qpos_array)[:seq_length]
            action_array = np.array(self.action_array)
                
            f.create_dataset("color", data=color_array[:-discard_end_length])
            f.create_dataset("depth", data=depth_array[:-discard_end_length])
            f.create_dataset("cloud", data=cloud_array[:-discard_end_length])
            f.create_dataset("env_qpos_proprioception", data=env_qpos_array[:-discard_end_length])
            f.create_dataset("action", data=action_array[:-discard_end_length])

        cprint(f"color shape: {color_array.shape}", "yellow")
        cprint(f"depth shape: {depth_array.shape}", "yellow")
        cprint(f"cloud shape: {cloud_array.shape}", "yellow")
        cprint(f"action shape: {action_array.shape}", "yellow")
        cprint(f"env_qpos shape: {env_qpos_array.shape}", "yellow")
        cprint(f"save data at step: {seq_length} in {record_file_name}", "yellow")
        cprint(f"Final saved file: {record_file_name}", "green")
        cprint("Program finished.", "green")



def build_argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Collect data with ARX controller + Realsense")
    parser.add_argument("model", type=str, help="ARX arm model, e.g., X5 or L5")
    parser.add_argument("interface", type=str, help="CAN interface name, e.g., can0")
    parser.add_argument("--fps", type=int, default=50, help="capture frequency (Hz)")
    parser.add_argument("--length", type=int, default=2000, help="number of steps to record")
    parser.add_argument("--demo_dir", type=str, default="demo_dir/raw_data", help="directory to save H5")
    parser.add_argument("--resize", type=int, default=224, help="color/depth resize square size")
    parser.add_argument("--save_img", type=int, default=1, help="save color images (1/0)")
    parser.add_argument("--save_depth", type=int, default=1, help="save depth maps (1/0)")
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
        resize_hw=args.resize,
        save_img=bool(args.save_img),
        save_depth=bool(args.save_depth),
    )

    await collector.run()


def main():
    import argparse
    parser = build_argparser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
