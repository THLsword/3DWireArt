#!/bin/bash

model_name='rabbit'
template=sphere24
# templates: [donut, cube24, sphere24, sphere54]
output_filename=outputs/${model_name}

# python src/post_perceptual.py \
# --model_path=./data/models/${model_name} \
# --template_path=data/templates/${template} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/prep_outputs/train_outputs \
# --alpha_value=0.25 \
# --object_curve_num=48 \
# --mv_thresh=0.1

python src/post_perceptual.py \
--model_path=./data/models/${model_name} \
--template_path=data/templates/${template} \
--output_path=./${output_filename} \
--prep_output_path=./${output_filename}/prep_outputs/train_outputs \
--alpha_value=0.25 \
--object_curve_num=35 \
--mv_thresh=0.1

# python src/post_perceptual.py \
# --model_path=./data/models/${model_name} \
# --template_path=data/templates/${template} \
# --output_path=./${output_filename} \
# --prep_output_path=./${output_filename}/prep_outputs/train_outputs \
# --alpha_value=0.25 \
# --object_curve_num=25 \
# --mv_thresh=0.1