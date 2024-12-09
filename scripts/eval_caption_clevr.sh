#!/bin/bash

DATA_PATH=/home/pooyan/clevr_data/
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.1.10 --master_port 29510 main_task_caption.py \
--do_eval \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir ckpts/onTest_eval \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype clevr \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/16 \
--init_model ckpts/clevr-caption-sup-vit-b16/pytorch_model.bin.13
