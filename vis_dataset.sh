#!/bin/bash

# bash vis_dataset.sh

dataset_path=/home/zishang/python_workspace/Humanoid-Teleoperation/demo_dir/training_data_grasp

start=0
end=-1
stride=5
delay=0.05
point_size=5.0
window_name="Zarr Point Clouds"
loop_flag=1

cd teleoperation
python visualize_zarr_point_clouds.py "${dataset_path}" \
    --start "${start}" \
    --end "${end}" \
    --stride "${stride}" \
    --delay "${delay}" \
    --point-size "${point_size}" \
    --window-name "${window_name}" \
    --loop "${loop_flag}"
