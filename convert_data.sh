# bash convert_data.sh


save_img=0
save_depth=0


demo_path=/home/zishang/python-ws/Humanoid-Teleoperation/demo_dir/raw_data
save_path=/home/zishang/python-ws/Humanoid-Teleoperation/demo_dir/training_data_bottle3rd

cd humanoid_teleoperation/scripts
python convert_demos.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
