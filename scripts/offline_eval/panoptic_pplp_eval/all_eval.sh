#!/bin/bash

# set -e
# set -x

# Sort by step
folders=$(ls ./$1/ | sort -V)

for folder in $folders
do
	echo "$folder" | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /home/trinhle/Downloads/cam_lidar_mocap_test_set/raw_static_segway_20181031/training/label_2 $1/$folder | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /mnt/fcav/projects/bodypose2dsim/160422_ultimatum1/training/label_2 $1/$folder | tee -a ./results_$1.txt
	./evaluate_object_3d_offline /home/trinhle/Panoptic/projects/bodypose2dsim/160422_ultimatum1/training/label_2 $1/$folder | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /mnt/fcav/projects/bodypose2dsim/panoptic_kittized_dataset/training/label_2 $1/$folder | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /mnt/fcav/projects/bodypose2dsim/160226_haggling1/training/label_2 $1/$folder | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /home/trinhle/Panoptic/projects/bodypose2dsim/160226_haggling1/training/label_2 $1/$folder | tee -a ./results_$1.txt
	# ./evaluate_object_3d_offline /mnt/fcav/projects/bodypose2dsim/171204_pose3/training/label_2 $1/$folder | tee -a ./results_$1.txt

done
