#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque 
import imageio
import pyrealsense2 as rs  # type: ignore
from multiprocessing import Process, Queue
import time
import multiprocessing
from typing import Optional
multiprocessing.set_start_method('fork')

np.printoptions(3, suppress=True)

def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

def init_given_realsense_L515(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        # L515
        h, w = 768, 1024
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        # L515
        h, w = 540, 960
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        # for L515
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        
        # set min distance
        # for L515
        # depth_sensor.set_option(rs.option.min_distance, 0.05)
        depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)

        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None

def init_given_realsense_D455(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D455
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        # h, w = 720, 1280
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None

def init_given_realsense_D435(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        #     Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        
        # D455
        h, w = 720, 1280
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        
        h, w = 720, 1280
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None




def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]


def uniform_sample_pcd(pcd: np.ndarray, k: int) -> np.ndarray:
    """简单的均匀采样/填充函数：
    若点数 >= k，则随机有放回采样 k 个点；若点数 < k，则末尾补零点直到 k"""
    n, dim = pcd.shape
    if k <= 0:
        return np.zeros((0, dim), dtype=pcd.dtype)
    if n >= k:
        idx = np.random.choice(n, k, replace=True)  # 与原片段保持 replace=True
        return pcd[idx]
    else:
        num_pad = k - n
        pad_points = np.zeros((num_pad, dim), dtype=pcd.dtype)
        return np.concatenate([pcd, pad_points], axis=0)


class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale
        
class SingleVisionProcess(Process):
    def __init__(self, device, queue,
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=0,
                num_points=2048,
                z_far=1.0,
                z_near=0.1,
                use_grid_sampling=True,
                img_size=384,
                yolo_config: Optional[dict] = None) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode
            
        self.use_grid_sampling = use_grid_sampling

  
        self.resize = False
        # self.height, self.width = 512, 512
        self.height, self.width = img_size, img_size
        
        # point cloud params
        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points

        # segmentation config/provider
        self.mask_provider = None
        # use fg_ratio from yolo_config if provided, else default 0.7
        self.fg_ratio = float(yolo_config.get('fg_ratio', 0.7)) if yolo_config else 0.7
        # optional provider config to build YOLO inside child process (GPU-safe)
        self.yolo_config = yolo_config
    
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()
        seg_mask = None

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
    
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            clip_lower =  0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high
            
            if self.enable_pointcloud:
                # Optional: run external segmentation to get a binary mask on the color frame
                if self.mask_provider is not None:
                    t_start = time.perf_counter()
                    seg_mask = self.mask_provider(color_frame)
                    t_end = time.perf_counter()
                    print(f"seg_mask generation took {(t_end - t_start) * 1000:.2f} ms")

                # Nx6
                point_cloud_frame = self.create_colored_point_cloud(
                    color_frame,
                    depth_frame,
                    far=self.z_far,
                    near=self.z_near,
                    num_points=self.num_points,
                    seg_mask=seg_mask,
                    fg_ratio=self.fg_ratio,
                )
            else:
                point_cloud_frame = None
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)
        
        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.enable_depth:
                depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_frame, depth_frame, point_cloud_frame, seg_mask


    def run(self):
        device_name = "L515"
        if device_name == "L515":
            init_given_realsense = init_given_realsense_L515
        elif device_name == "D435":
            init_given_realsense = init_given_realsense_D435
        elif device_name == "D455":
            init_given_realsense = init_given_realsense_D455
        
        if self.mask_provider is None and self.yolo_config is not None:
            from ultralytics import YOLO  # type: ignore
            model_path = self.yolo_config.get('model_path', 'yolov8n-seg.pt')
            device = self.yolo_config.get('device', 'cuda:0')
            classes = self.yolo_config.get('classes', None)
            conf = float(self.yolo_config.get('conf', 0.25))

            yolo_model = YOLO(model_path)
            yolo_model.to(device)

            mode = str(self.yolo_config.get('mode', 'segment')).lower()
            if mode == 'detect':
                from yolo_detection import YoloDetection
                self.mask_provider = YoloDetection(model=yolo_model, target_classes=classes, conf=conf)
                print('[INFO] Child process YOLO detection-mask initialized on', device)
            else:
                from yolo_segmentation import YoloSegmentation
                self.mask_provider = YoloSegmentation(model=yolo_model, target_classes=classes, conf=conf)
                print('[INFO] Child process YOLO segmentation initialized on', device)

            self.fg_ratio = float(self.yolo_config.get('fg_ratio', self.fg_ratio))

        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(self.device, 
                    enable_rgb=self.enable_rgb, enable_depth=self.enable_depth,
                    enable_point_cloud=self.enable_pointcloud,
                    sync_mode=self.sync_mode)

        while True:
            color_frame, depth_frame, point_cloud_frame, seg_mask = self.get_vision()
            self.queue.put([color_frame, depth_frame, point_cloud_frame, seg_mask])
            time.sleep(0.05)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

    def create_colored_point_cloud(self, color, depth, far=1.0, near=0.1, num_points=10000, seg_mask=None, fg_ratio=0.7):
        assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
        
        # Clip points based on depth
        valid_mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        cloud = cloud[valid_mask]
        color = color.reshape([-1, 3])
        color = color[valid_mask]

        # flatten segmentation mask and sync with valid_mask
        seg_flat = None
        if seg_mask is not None:
            seg_flat = seg_mask.reshape(-1)[valid_mask]
            seg_flat = seg_flat.astype(bool)


        colored_cloud = np.hstack([cloud, color.astype(np.float32)])

        if seg_flat is None:
            # 先体素下采样，然后均匀采样
            if self.use_grid_sampling and colored_cloud.shape[0] > 0:
                colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
            colored_cloud = uniform_sample_pcd(colored_cloud, num_points)
        else:
            # foreground/background separate strategy
            fg_cloud = colored_cloud[seg_flat]
            bg_cloud = colored_cloud[~seg_flat]
            num_fg_cloud = int(num_points * fg_ratio)
            num_bg_cloud = num_points - num_fg_cloud

            if self.use_grid_sampling:
                if fg_cloud.shape[0] > 0:
                    fg_cloud = grid_sample_pcd(fg_cloud, grid_size=0.005)
                if bg_cloud.shape[0] > 0:
                    bg_cloud = grid_sample_pcd(bg_cloud, grid_size=0.005)

            fg_sel = uniform_sample_pcd(fg_cloud, num_fg_cloud) if num_fg_cloud > 0 else np.zeros((0, 6), dtype=np.float32)
            bg_sel = uniform_sample_pcd(bg_cloud, num_bg_cloud) if num_bg_cloud > 0 else np.zeros((0, 6), dtype=np.float32)
            colored_cloud = np.concatenate([fg_sel, bg_sel], axis=0)
        
        # shuffle
        np.random.shuffle(colored_cloud)
        return colored_cloud



class MultiRealSense(object):
    def __init__(self, use_front_cam=True, use_right_cam=False,
                 front_cam_idx=0, right_cam_idx=1, 
                 front_num_points=4096, right_num_points=1024,
                 front_z_far=1.0, front_z_near=0.1,
                 right_z_far=0.5, right_z_near=0.01,
                 use_grid_sampling=True, 
                 img_size=512,
                 yolo_config: Optional[dict] = None,
                 ):

        self.devices = get_realsense_id()
    
        self.front_queue = Queue(maxsize=3)
        self.right_queue = Queue(maxsize=3)

      
        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        if use_front_cam:
            self.front_process = SingleVisionProcess(self.devices[front_cam_idx], self.front_queue,
                            enable_rgb=True, enable_depth=True, enable_pointcloud=True, sync_mode=1,
                            num_points=front_num_points, z_far=front_z_far, z_near=front_z_near, 
                            use_grid_sampling=use_grid_sampling, img_size=img_size,
                            yolo_config=yolo_config)
        if use_right_cam:
            self.right_process = SingleVisionProcess(self.devices[right_cam_idx], self.right_queue,
                    enable_rgb=True, enable_depth=True, enable_pointcloud=True, sync_mode=1,
                        num_points=right_num_points, z_far=right_z_far, z_near=right_z_near, 
                        use_grid_sampling=use_grid_sampling, img_size=img_size,
                        yolo_config=yolo_config)


        if use_front_cam:
            self.front_process.start()
            print("front camera start.")

        if use_right_cam:
            self.right_process.start()
            print("right camera start.")

        self.use_front_cam = use_front_cam
        self.use_right_cam = use_right_cam
        
        
    def __call__(self):  
        cam_dict = {}
        if self.use_front_cam:  
            front_color, front_depth, front_point_cloud, front_mask = self.front_queue.get()
            cam_dict.update({'color': front_color, 'depth': front_depth, 'point_cloud':front_point_cloud, 'mask': front_mask})
      
        if self.use_right_cam: 
            right_color, right_depth, right_point_cloud, right_mask = self.right_queue.get()
            cam_dict.update({'right_color': right_color, 'right_depth': right_depth, 'right_point_cloud':right_point_cloud, 'right_mask': right_mask})
        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            self.front_process.terminate()
        if self.use_right_cam:
            self.right_process.terminate()


    def __del__(self):
        self.finalize()
        

if __name__ == "__main__":
    # 配置在子进程内构建 YOLO
    YOLO_CONFIG = {
        'model_path': 'yolo11n.pt',
        'device': 'cuda:0',    # 改为你的 GPU，例如 'cuda:0'
        'classes': {39},       # COCO 类别集合；None 表示不过滤
        'conf': 0.25,
        'fg_ratio': 0.8,       # 前景点比例，仅在使用分割或检测掩码时影响前景/背景分配
        'mode': 'detect',      # 只能是 'detect' 或 'segment'；'detect' 用检测框生成掩码，'segment' 用分割掩码
    }

    cam = MultiRealSense(use_right_cam=False, front_num_points=10000,
                         use_grid_sampling=True, img_size=1024,
                         front_z_far=1.0, front_z_near=0.2,
                         yolo_config=YOLO_CONFIG)

    from open3d_viz import AsyncPointCloudViewer
    viewer = AsyncPointCloudViewer(
        width=960,
        height=720,
        point_size=2.0,
        queue_size=1,
        window_name="RealSense Live Point Cloud",
    )

    while True:
        # measure and print the time taken by cam()
        t0 = time.perf_counter()
        out = cam()
        t1 = time.perf_counter()
        print(f"cam() took {(t1 - t0) * 1000:.2f} ms")
        # visualize point cloud in Open3D
        pc = out.get("point_cloud", None)
        if pc is not None and pc.size:
            viewer.update(pc)
        else:
            time.sleep(0.01)

        # visualize color + mask overlay
        color = out.get('color', None)
        if color is not None:
            bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            mask = out.get('mask', None)
            # inline visualize_mask to avoid extra imports
            if mask is not None:
                m = mask.astype(bool)
                o = bgr.copy()
                o[m] = (0, 255, 0)
                vis = cv2.addWeighted(o, 0.5, bgr, 0.5, 0)
            else:
                vis = bgr
            cv2.imshow('Color+Mask', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    viewer.close()
    cam.finalize()
    cv2.destroyAllWindows()