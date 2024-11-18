#! /bin/bash

DATASET_NAME=$1
SCENE_NAME=$2
ARGS=$3

bash bash/train.sh $DATASET_NAME $SCENE_NAME $ARGS
bash bash/render.sh $DATASET_NAME $SCENE_NAME $ARGS
# bash bash/eval.sh $DATASET_NAME $SCENE_NAME