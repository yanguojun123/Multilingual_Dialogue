#!/bin/bash
port=$(($(date +%N)%30000))

# Set-up the environment.
cd $2
python -m torch.distributed.launch --nproc_per_node=1 --master_port $port $4.py \
--source_language=$5 \
--target_language=$6 \
--train_file_name=$7 \
--val_file_name=$8 \
--test_file_name=$9 \
--save_file_name=$3.json \
--writer_dir=tensorboard_$3/ \
--dialogue_model_output_path=$3_model/ \
--eval_all_checkpoints \
--change_parameter \
${10}