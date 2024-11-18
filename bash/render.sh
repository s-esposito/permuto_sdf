#! /bin/bash

DATASET_NAME=$1
SCENE_NAME=$2
ARGS=$3

python ./permuto_sdf_py/experiments/evaluation/create_my_images.py \
--dataset $DATASET_NAME \
--scene $SCENE_NAME \
--comp_name comp_3 \
$ARGS