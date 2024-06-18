#! /bin/bash

DATASET_NAME=$1
SCENE_NAME=$2

python ./permuto_sdf_py/experiments/evaluation/evaluate_psnr.py \
--dataset $DATASET_NAME \
--scene $SCENE_NAME