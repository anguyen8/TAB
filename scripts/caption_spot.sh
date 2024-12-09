#!/bin/bash

export JAVA_HOME="your_jdk/"
export PATH=$JAVA_HOME/bin:$PATH

DATA_PATH=your_data/std/data
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 main_task_caption.py \
--do_train \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/resized_images \
--output_dir ckpts/spot_caption\
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/pytorch_model.bin.spot