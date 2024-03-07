**MOLR with Decoder-only Backbones for DST and NLG tasks**

1. Prepare dataset and framework

We conduct LoRA PEFT with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) and follow their guidelines to prepare dataset ([prepare_llm_sft_dataset.ipynb](prepare_llm_sft_dataset.ipynb) and [prepare_llm_sft_nlg.ipynb](prepare_llm_sft_nlg.ipynb)) and register the datasets in `LLaMA-Factory/data/dataset_info.json`
We uploaded the following processed datasets to huggingface:
- https://huggingface.co/datasets/Jiahuan/dst_en
- https://huggingface.co/datasets/Jiahuan/dst_de
- https://huggingface.co/datasets/Jiahuan/dst_it
- https://huggingface.co/datasets/Jiahuan/dst_mix_en_de_it
- https://huggingface.co/datasets/Jiahuan/nlg_en
- https://huggingface.co/datasets/Jiahuan/nlg_de
- https://huggingface.co/datasets/Jiahuan/nlg_it
- https://huggingface.co/datasets/Jiahuan/nlg_mix_en_de_it

2. LoRA fine-tuning on clusters such as Snellius
- DST task
```shell
# Monolingual
sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_en -t llama2 -e en -l q_proj,v_proj -n 60 -b 8
# Multilingual with MOLR
sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_mix -t llama2 -e mix -l q_proj,v_proj -n 10 -b 8
```
- NLG task
```shell
# Monolingual
sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_en -t llama2 -e en -l q_proj,v_proj -n 60 -b 8
# Multilingual with MOLR
sh molr_run_snellius.sh -r train -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_mix -t llama2 -e mix -l q_proj,v_proj -n 10 -b 8
```
3. Inference and evaluation on our lab's gpus
- DST Task
```shell
# Monolingual
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_en -t llama2 -e en -p en -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_en_2024-02-19-11-31-42
python evaluate_molr_dst_acc_v2.py --target_language en --result_file_name /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_en_2024-02-19-11-31-42/generated_predictions.jsonl
# Multilingual with MOLR
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_en -t llama2 -e en -p en -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400
python evaluate_molr_dst_acc_v2.py --target_language en --result_file_name /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400/en/generated_predictions.jsonl
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_de -t llama2 -e de -p de -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400
python evaluate_molr_dst_acc_v2.py --target_language de --result_file_name /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400/de/generated_predictions.jsonl
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_dst_it -t llama2 -e it -p it -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400
python evaluate_molr_dst_acc_v2.py --target_language it --result_file_name /media/Blue2TB3/jpei/molr-finetune-LLaMAFactory/LLaMA2-7B-Chat/lora/train_mix_2024-02-21-16-16-34/checkpoint-21400/it/generated_predictions.jsonl
```
- NLG task
```shell
# Monolingual
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_en -t llama2 -e en -p en -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/Snellius/save/molr-finetune-LLaMAFactory/nlg/meta-llama/Llama-2-7b-chat-hf/lora/train_en_20240229_185522
# Multilingual with MOLR
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_en -t llama2 -e en -p en -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/Snellius/save/molr-finetune-LLaMAFactory/nlg/meta-llama/Llama-2-7b-chat-hf/lora/train_mix_20240301_145041
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_de -t llama2 -e de -p de -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/Snellius/save/molr-finetune-LLaMAFactory/nlg/meta-llama/Llama-2-7b-chat-hf/lora/train_mix_20240301_145041
sh molr_run.sh -r test -m meta-llama/Llama-2-7b-chat-hf -d molr_nlg_it -t llama2 -e it -p it -l q_proj,v_proj -b 4 -a /media/Blue2TB3/jpei/Snellius/save/molr-finetune-LLaMAFactory/nlg/meta-llama/Llama-2-7b-chat-hf/lora/train_mix_20240301_145041
```