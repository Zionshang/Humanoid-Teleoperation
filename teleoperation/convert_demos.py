import os
import argparse
import pickle
import numpy as np
import random
import time
from termcolor import colored
import h5py
import zarr
from termcolor import cprint
from tqdm import tqdm
import argparse
from pcd_downsampling import grid_sample_pcd, color_weighted_downsample, random_uniform_downsample


def _parse_color_arg(color_str: str):
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"color_target must have 3 comma-separated values, got '{color_str}'")
    return tuple(float(p) for p in parts)


def convert_dataset(args):
    demo_dir = args.demo_dir
    save_dir = args.save_dir
    
    save_img = args.save_img
    save_depth = args.save_depth
    use_voxel = bool(args.voxel_sampling)
    voxel_size = args.voxel_size
    use_color_weight = bool(args.color_weight_sampling)
    color_temperature = args.color_temperature
    color_target = _parse_color_arg(args.color_target)
    num_points = args.num_points
    
    # create dir to save demonstrations
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".h5")]
    demo_files = sorted(demo_files)
    
    
    total_count = 0
    color_arrays = []
    depth_arrays = []
    cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    
    
    for demo_file in demo_files:
        # load file (h5)
        file_name = os.path.join(demo_dir, demo_file)
        print("process:", file_name)

        with h5py.File(file_name, "r") as data:


            if save_img:
                color_array = data["color"][:]
                
            if save_depth:
                depth_array = data["depth"][:]
              
            cloud_array = data["cloud"][:]
            action_array = data["action"][:]
            proprioception_array = data["env_qpos_proprioception"][:]
                
                
            # to list
            length = len(cloud_array)
            if save_img:
                color_array = [color_array[i] for i in range(length)]
              
            if save_depth:
                depth_array = [depth_array[i] for i in range(length)]
               
            
            new_cloud_array = []
            for i in range(length):
                old_cloud = cloud_array[i]
                if use_voxel:
                    old_cloud = grid_sample_pcd(old_cloud, grid_size=voxel_size)
                    print("num points after voxel downsample:", old_cloud.shape[0])

                if use_color_weight:
                    new_cloud = color_weighted_downsample(
                        old_cloud,
                        num_points=num_points,
                        target_color=color_target,
                        temperature=color_temperature,
                    )
                else:
                    new_cloud = random_uniform_downsample(old_cloud, num_points)
                
                new_cloud_array.append(new_cloud)
            
            cloud_array = new_cloud_array
         
            proprioception_array = [proprioception_array[i] for i in range(length)]
            action_array = [action_array[i] for i in range(length)]
         
    
        total_count += len(action_array)
        cloud_arrays.extend(cloud_array)
       
        if save_img:
            color_arrays.extend(color_array)
  
        if save_depth:
            depth_arrays.extend(depth_array)
            
        state_arrays.extend(proprioception_array)
        action_arrays.extend(action_array)
        episode_ends_arrays.append(total_count)

    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    if save_img:
        color_arrays = np.stack(color_arrays, axis=0)
        if color_arrays.shape[1] == 3: # make channel last
            color_arrays = np.transpose(color_arrays, (0,2,3,1))
       
    if save_depth:
        depth_arrays = np.stack(depth_arrays, axis=0)
       
        
    state_arrays = np.stack(state_arrays, axis=0)
    cloud_arrays = np.stack(cloud_arrays, axis=0)
  
        
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    single_size = 500
    state_chunk_size = (single_size, state_arrays.shape[1])
    point_cloud_chunk_size = (single_size, cloud_arrays.shape[1], cloud_arrays.shape[2])
    action_chunk_size = (single_size, action_arrays.shape[1])
    if save_img:
        img_chunk_size = (single_size, color_arrays.shape[1], color_arrays.shape[2], color_arrays.shape[3])
        zarr_data.create_dataset('img', data=color_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        
    if save_depth:
        depth_chunk_size = (single_size, depth_arrays.shape[1], depth_arrays.shape[2])
        zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        
    
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
      
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    # print shape
    if save_img:
        cprint(f'color shape: {color_arrays.shape}, range: [{np.min(color_arrays)}, {np.max(color_arrays)}]', 'green')
    if save_depth:
        cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'cloud shape: {cloud_arrays.shape}, range: [{np.min(cloud_arrays)}, {np.max(cloud_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # count file size
    total_size = 0
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))
    cprint(f"Total size: {total_size/1e6} MB", "green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_img", type=int)
    parser.add_argument("--save_depth", type=int)
    parser.add_argument("--voxel_sampling", type=int, default=0, help="Enable voxel grid sampling before saving.")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for grid sampling.")
    parser.add_argument("--color_weight_sampling", type=int, default=0, help="Enable color weighted sampling.")
    parser.add_argument("--color_target", type=str, default="255,255,0", help="Target RGB color for weighted sampling, e.g. '255,255,0'.")
    parser.add_argument("--color_temperature", type=float, default=50.0, help="Sigma for Gaussian color weighting.")
    parser.add_argument("--num_points", type=int, default=10000, help="Number of points to keep per cloud.")
    
    
    args = parser.parse_args()
    
    convert_dataset(args)
    
