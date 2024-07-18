#!/bin/bash

DATA_PATH=your_data/openImages-C/data
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--do_train \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=64 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/ckpt_openC_b32 \
--lr 1e-3 \
--max_words 32 \
--batch_size_val 128 \
--datatype open \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/pretrained/pytorch_model.bin.clevr
