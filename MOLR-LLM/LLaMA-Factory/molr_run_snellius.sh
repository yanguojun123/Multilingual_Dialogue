#!/bin/bash
#DATA_DIR="/media/Blue2TB3/jpei/molr-finetune-LLaMAFactory"
#DATA_DIR="/scratch-local/jpei/molr-finetune-LLaMAFactory" # Delete automatically by 6 days
DATA_DIR="save/molr-finetune-LLaMAFactory" # Delete automatically by 6 days
## parse bash parameters
while getopts r:m:a:l:d:t:e:n:b:p: option
do
   case "${option}"  in
                r) MODE=${OPTARG};;    # train/test
                m) MODEL=${OPTARG};;   # model name
                a) ADAPTER=${OPTARG};;   # adapter folder
                l) LORA=${OPTARG};;   # lora target
                d) DATA=${OPTARG};;    # dataset name
                t) TEMP=${OPTARG};;    # template name
                e) EXP=${OPTARG};;     # exp name
                n) NEPOCH=${OPTARG};;     # number of epoch
                b) BATCH=${OPTARG};;     # number of epoch
                p) PARAMS=${OPTARG};;  # other parameters
   esac
done

# sh molr_run.sh -r train -m ${MODEL} -d ${DATA} -e ${EXP}

## Construct log dir
TS=$(date +"%Y%m%d_%H%M%S")
log_dir=${ROOT}/log/${MODEL}_${DATA}_${EXP}/${TS}
if [ ! -d ${log_dir}  ];then
  mkdir -p ${log_dir}
fi

echo "sh run.sh -r ${MODE} -m ${MODEL} -d ${DATA} -t ${TEMP} -e ${EXP} -l ${LORA} -n ${NEPOCH} -b ${BATCH} -a ${ADAPTER}"

#### Training Settings
### DST Task
## 1. Llama2
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_en -t llama2 -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_de -t llama2 -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_it -t llama2 -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_mix -t llama2 -e mix -l q_proj,v_proj -n 10 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_all -t llama2 -e all -l q_proj,v_proj -n 30 -b 8
## 2. Bloom
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_dst_en -t default -e en -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_dst_de -t default -e de -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_dst_it -t default -e it -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_dst_mix -t default -e mix -l query_key_value -n 10 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_dst_all -t default -e all -l query_key_value -n 30 -b 8
## 3. OpenChat3.5
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_dst_en -t openchat -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_dst_de -t openchat -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_dst_it -t openchat -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_dst_mix -t openchat -e mix -l q_proj,v_proj -n 10 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_dst_all -t openchat -e all -l q_proj,v_proj -n 30 -b 8
### NLG Task
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_en -t llama2 -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_de -t llama2 -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_it -t llama2 -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_mix -t llama2 -e mix -l q_proj,v_proj -n 10 -b 8
# sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_all -t llama2 -e all -l q_proj,v_proj -n 30 -b 8
## 2. Bloom
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_nlg_en -t default -e en -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_nlg_de -t default -e de -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_nlg_it -t default -e it -l query_key_value -n 60 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_nlg_mix -t default -e mix -l query_key_value -n 10 -b 8
# sh molr_run_snellius.sh -r train -m bigscience/bloom-7b1 -d molr_nlg_all -t default -e all -l query_key_value -n 30 -b 8
## 3. OpenChat3.5
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_nlg_en -t openchat -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_nlg_de -t openchat -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_nlg_it -t openchat -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_nlg_mix -t openchat -e mix -l q_proj,v_proj -n 10 -b 8
# sh molr_run_snellius.sh -r train -m openchat/openchat_3.5 -d molr_nlg_all -t openchat -e all -l q_proj,v_proj -n 30 -b 8

#### Test Settings
# sh molr_run_snellius.sh -r test -m bigscience/bloom-7b1 -d molr_dst_en -t default -e en -p en -l q_proj,v_proj -b 4 -a save/molr-finetune-LLaMAFactory/bigscience/bloom-7b1/lora/train_en_20240227_170832
# sh molr_run_snellius.sh -r test -m bigscience/bloom-7b1 -d molr_dst_de -t default -e de -p de -l q_proj,v_proj -b 4 -a save/molr-finetune-LLaMAFactory/bigscience/bloom-7b1/lora/train_de_20240227_170838
# sh molr_run_snellius.sh -r test -m bigscience/bloom-7b1 -d molr_dst_it -t default -e it -p it -l q_proj,v_proj -b 4 -a save/molr-finetune-LLaMAFactory/bigscience/bloom-7b1/lora/train_it_20240227_170844


sbatch <<EOT
#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name="${EXP}_${MODEL}"MolrSft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3-00:00:00
#SBATCH --output="${EXP}_${MODEL}"MolrSft_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# Activate your environment
source activate dl2023

pip install -r requirements.txt

wandb login 8d481fbff8b334b14b756f0ac236f5496e91b448

which python
ROOT=$(dirname $(realpath ${0}))
export PYTHONPATH=${ROOT}

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path "${MODEL}" \
    --dataset "${DATA}" \
    --split "${MODE}" \
    --template "${TEMP}" \
    --lora_target "${LORA}" \
    --num_train_epochs "${NEPOCH}" \
    --output_dir "${DATA_DIR}"/"${MODEL}"/lora/"${MODE}_${EXP}_${TS}" \
    --molr_exp "${EXP}" \
    --dataset_dir data \
    --do_train True \
    --stage sft \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --do_eval True \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --val_size 0.2 \
    --cutoff_len 1024 \
    --learning_rate 1.5e-04 \
    --per_device_train_batch_size "${BATCH}" \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 2000 \
    --fp16 True \
    --plot_loss True
EOT