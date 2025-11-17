# bash convert_data.sh


save_img=0
save_depth=0


demo_path=/home/zishang/python_workspace/Humanoid-Teleoperation/demo_dir/raw_data_bottle3rd_raw_pcd
save_path=/home/zishang/python_workspace/Humanoid-Teleoperation/demo_dir/training_data_bottle3rd_raw_pcd

voxel_sampling=1
voxel_size=0.002
color_weight_sampling=1
color_target="255,255,0"
color_temperature=20.0
num_points=4096

cd teleoperation
python convert_demos.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
                                --voxel_sampling ${voxel_sampling} \
                                --voxel_size ${voxel_size} \
                                --color_weight_sampling ${color_weight_sampling} \
                                --color_target ${color_target} \
                                --color_temperature ${color_temperature} \
                                --num_points ${num_points}
