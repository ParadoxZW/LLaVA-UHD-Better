#!/bin/bash

HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-uhd-v1.5-7b-lora"
CKPT="${MODEL_NAME}_1"
SPLIT="mmbench_dev_20230712_2e"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_uhd/eval/model_vqa_mmbench.py \
        --model-path ./checkpoints/$MODEL_NAME \
        --model-base /data/llm_common/vicuna-7b-v1.5 \
        --question-file ./playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv \
        --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done


wait

output_file=./playground/data/eval/mmbench_cn/answers/$SPLIT/$CKPT/${MODEL_NAME}_cn.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/mmbench_cn/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


mkdir -p ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT/${CKPT}_cn

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv \
    --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT/$CKPT \
    --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT/${CKPT}_cn \
    --experiment ${MODEL_NAME}_cn
