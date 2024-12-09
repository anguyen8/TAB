#!/bin/bash

DATA_PATH=/home/pooyan/open-images-c/
python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.4 --master_port 29504 main_task_caption.py \
--do_eval \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH} \
--output_dir /home/pooyan/CAB-CC/open_eval_test_32 \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype open \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model ckpts/open-c-caption-sup-vit-b32/pytorch_model.bin.12
