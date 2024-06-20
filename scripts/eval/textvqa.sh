#!/bin/bash

HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-uhd-v1.5-7b-lora2"
CKPT="${MODEL_NAME}_1"
SPLIT="llava_uhd_v1"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava_uhd.eval.model_vqa_loader \
        --model-path ./checkpoints/$MODEL_NAME \
        --model-base /data/llm_common/llava-v1.5-7b \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder /data/ouyangxc/data/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done


wait

output_file=./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done



python -m llava_uhd.eval.eval_textvqa \
    --annotation-file /data/ouyangxc/data/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file
