#!/bin/bash
CUR_DATA_DIR=$DATA_DIR
port=$(($(date +%N)%30000))
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$4
#SBATCH -o output/out_$4
#SBATCH -e output/err_$4
#SBATCH -p si
#SBATCH --nodelist $1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

# Set-up the environment.
source activate
conda activate mul
cd $2
# set PYTHONPATH=./
# Start the experiment.
#CUDA_VISIBLE_DEVICES=0 
python -m torch.distributed.launch --nproc_per_node=1 --master_port $port $3.py \
--source_language '$5' \
--target_language '$6' \
--train_file_name '$7' \
--val_file_name '$8' \
--test_file_name '$9' \
--save_file_name '$4.json' \
--writer_dir 'tensorboard_$4/' \
--dialogue_model_output_path '$4_model/' \
--eval_all_checkpoints \
--change_parameter \
${10}
#--pretrained_model '../output/model/MT5_mapping_hidden_ende_de_adam_test_model/model_epoch46/'
#Example: sbatch run_mt5.sh $name $gpu $gpu_device_number $train_path  $val_path $test_path $language $prefix(if it's true, input will add prefix)
#Example: sbatch run_mt5.sh en_test gpu04 0 belief_en_train.txt belief_en_val.txt belief_en_test.txt en --prefix
EOT