#!/bin/bash

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="openbmb/MiniCPM-V-2_6"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="path/to/trainging_data"
EVAL_DATA="path/to/test_data"

# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
LLM_TYPE="qwen" 
MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 3 \
    --output_dir output/model \
    --logging_dir output/model \
    --max_line_res 1120 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing false  \
    --deepspeed ds_config_zero2.json  \
    --report_to "tensorboard" \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 16 \
