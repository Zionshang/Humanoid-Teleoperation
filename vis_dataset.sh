# bash scripts/vis_dataset.sh

dataset_path=/home/zishang/python-ws/Humanoid-Teleoperation/demo_dir/training_data_bottle

vis_cloud=1
python ./humanoid_teleoperation/scripts/vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 0 \
                    --downsample 1 \