#!/bin/bash

DATA_PATH=/home/pooyan/VIRAT-STD
python -m torch.distributed.launch --nproc_per_node=1 --master_port=5557 main_task_caption.py \
--do_eval \
--num_thread_reader=4 \
--epochs=50 \
--batch_size=16 \
--n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/resized_images \
--output_dir ckpts/ckpt_spot_caption_eval_test_32 \
--lr 1e-4 \
--max_words 32 \
--batch_size_val 32 \
--datatype spot \
--coef_lr 1e-3 \
--freeze_layer_num 0 \
--linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
--init_model /home/pooyan/CAB-CC/ckpts/ckpt_spot_caption_32/pytorch_model.bin.10
