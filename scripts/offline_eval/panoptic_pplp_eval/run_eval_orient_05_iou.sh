#!/bin/bash
set -e

cd $1
echo "$3" | tee -a ./$4_results_orient_05_iou_$2.txt

./evaluate_orient_offline_05_iou /home/trinhle/Panoptic/projects/bodypose2dsim/160422_ultimatum1/training/label_2 $2/$3 | tee -a ./$4_results_orient_05_iou_$2.txt

cp $4_results_orient_05_iou_$2.txt $5
