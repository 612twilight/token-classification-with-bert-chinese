#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python run_pl_ner.py --data_dir ./data \
--labels label.txt \
--output_dir ./output_model \
--num_train_epochs 1 \
--train_batch_size 8 \
--gpus 1 \
--do_train \
--do_predict

# python run_pl_ner.py --data_dir ./data --labels label.txt --output_dir ./output_model --num_train_epochs 1 --train_batch_size 8 --do_train --do_predict