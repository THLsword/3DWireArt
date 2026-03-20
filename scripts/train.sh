#!/bin/bash

model_name='rabbit'
epoch=1500
learning_rate=0.0015
output_filename=outputs/${model_name}

python src/train.py  \
--model_path=./data/models/${model_name} \
--output_path=./${output_filename} \
--prep_output_path=./${output_filename}/prep_outputs/train_outputs \
--epoch=${epoch} \
--learning_rate=${learning_rate} \

