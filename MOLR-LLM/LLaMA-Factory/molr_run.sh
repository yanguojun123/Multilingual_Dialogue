#!/bin/bash
## Set-up the environment.
DATA_DIR="/media/Blue2TB3/jpei/molr-finetune-LLaMAFactory"
ROOT=$(dirname $(realpath ${0}))
N_GPU=$(nvidia-smi --list-gpus | wc -l)

which python
export PYTHONPATH=${ROOT}
#set OMP_NUM_THREADS=2

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

### Training
## 1. Llama2
# sh molr_run.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_en -t llama2 -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_de -t llama2 -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_it -t llama2 -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_mix -t llama2 -e mix -l q_proj,v_proj -n 60 -b 8
## 2. Bloom
# sh molr_run.sh -r train -m bigscience/bloom-7b1 -d molr_dst_en -t default -e en -l query_key_value -n 60 -b 4
# sh molr_run.sh -r train -m bigscience/bloom-7b1 -d molr_dst_de -t default -e de -l query_key_value -n 60 -b 4
# sh molr_run.sh -r train -m bigscience/bloom-7b1 -d molr_dst_it -t default -e it -l query_key_value -n 60 -b 4
# sh molr_run.sh -r train -m bigscience/bloom-7b1 -d molr_dst_mix -t default -e mix -l query_key_value -n 60 -b 4
## 3. OpenChat3.5
# sh molr_run.sh -r train -m openchat/openchat_3.5 -d molr_dst_en -t openchat -e en -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m openchat/openchat_3.5 -d molr_dst_de -t openchat -e de -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m openchat/openchat_3.5 -d molr_dst_it -t openchat -e it -l q_proj,v_proj -n 60 -b 8
# sh molr_run.sh -r train -m openchat/openchat_3.5 -d molr_dst_mix -t openchat -e mix -l q_proj,v_proj -n 60 -b 8

## Run code
if [ "${MODE}" = "train" ]; then
  CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
      --model_name_or_path "${MODEL}" \
      --dataset "${DATA}" \
      --split "${MODE}" \
      --template "${TEMP}" \
      --lora_target "${LORA}" \
      --num_train_epochs "${NEPOCH}" \
      --output_dir "${DATA_DIR}/${MODEL}/lora/${MODE}_${EXP}_${TS}" \
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
      --plot_loss True >> "${log_dir}/${MODE}.log" 2>&1 &
  ps -ef | grep "python src/train_bash.py"
  echo "LOG_DIR= ${log_dir}/${MODE}.log"
#  watch "tail -l ${log_dir}/${MODE}.log"
elif [ "${MODE}" = "test" ]; then
  # Inference
  CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
      --stage sft \
      --do_predict \
      --model_name_or_path "${MODEL}" \
      --adapter_name_or_path "${ADAPTER}" \
      --dataset "${DATA}" \
      --split "${MODE}" \
      --template "${TEMP}" \
      --finetuning_type lora \
      --output_dir "${ADAPTER}/${EXP}" \
      --per_device_eval_batch_size "${BATCH}" \
      --predict_with_generate \
      --fp16 >> "${ADAPTER}/${EXP}_${MODE}.log" 2>&1 &
elif [ "${MODE}" = "eval" ]; then
  # Evaluation
  python eval_molr_dst_acc.py --target_language "${PARAMS}" --result_file_name "${ADAPTER}/${EXP}/generated_predictions.jsonl"
else
  echo ""
fi
