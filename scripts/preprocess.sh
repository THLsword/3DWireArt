#!/bin/bash

model_name='rabbit'

output_filename=outputs/${model_name}

python src/preprocess.py \
--DATA_DIR=./data/models/${model_name} \
--RENDER_SAVE_DIR=./${output_filename}/prep_outputs/render_outputs \
--ALPHA_SAVE_DIR=./${output_filename}/prep_outputs/alpha_outputs \
--TRAIN_SAVE_DIR=./${output_filename}/prep_outputs/train_outputs \
--FILENAME=model_normalized_4096.npz \
--ALPHA_SIZE=30 \
--EXPAND_SIZE=1 