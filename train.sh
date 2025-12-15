#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
carla_dataset_type=Carla
carla_root_path_training_data=../datasets/carla_train

fastec_dataset_type=Fastec
fastec_root_path_training_data=../datasets/fastec_train

log_dir=../experiments/training_models/


cd deep_unroll_net

python train.py \
          --dataset_type=$carla_dataset_type \
          --dataset_root_dir=$carla_root_path_training_data \
          --log_dir=$log_dir \
          --net_type='netMiddle' \
          --lamda_perceptual=1 \
          --lamda_L1=10 \
          --lamda_gt_flow=0 \
          --lamda_flow_smoothness=0.1

python train.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_training_data \
          --log_dir=$log_dir \
          --net_type='netMiddle' \
          --lamda_perceptual=1 \
          --lamda_L1=10 \
          --lamda_gt_flow=0 \
          --lamda_flow_smoothness=0.1 
