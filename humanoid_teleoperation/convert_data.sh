# bash convert_data.sh


save_img=1
save_depth=0


demo_path=/home/jing/Humanoid-Teleoperation/humanoid_teleoperation/demo_dir
save_path=/home/jing/Humanoid-Teleoperation/humanoid_teleoperation/demo_dir/training_data_example

cd scripts
python convert_demos.py --demo_dir ${demo_path} \
                                --save_dir ${save_path} \
                                --save_img ${save_img} \
                                --save_depth ${save_depth} \
