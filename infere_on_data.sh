#!/usr/bin/env bash

DATA="/datagrid/personal/purkrmir/data/floorball_human_mesh_test/manual/all/"
vis_folder="maxsize"

# DATASET="3dpw"
DATASET="h36m"


OUT_DATA="/datagrid/personal/purkrmir/data/floorball_human_mesh_test/manual/all/METRO_"$vis_folder"_"$DATASET
MODEL="./models/metro_release/metro_"$DATASET"_state_dict.bin"

python \
    ./metro/tools/end2end_inference_bodymesh.py \
    --resume_checkpoint $MODEL \
    --image_file_or_path $DATA \
    --image_output $OUT_DATA

