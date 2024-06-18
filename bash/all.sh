#! /bin/bash

DATASET_NAME=$1
SCENE_NAME=$2

bash bash/train.sh $DATASET_NAME $SCENE_NAME
bash bash/render.sh $DATASET_NAME $SCENE_NAME
bash bash/eval.sh $DATASET_NAME $SCENE_NAME