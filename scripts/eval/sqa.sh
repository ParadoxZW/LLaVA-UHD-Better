#!/bin/bash

HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-uhd-v1.5-7b-lora"
CKPT="${MODEL_NAME}_1"
SPLIT="llava_uhd"

#/data/llm_common/vicuna-7b-v1.5 
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava_uhd/eval/model_vqa_loader.py   \
        --model-path ./checkpoints/$MODEL_NAME \
        --model-base /data/llm_common/vicuna-7b-v1.5 \
        --question-file ./playground/data/eval/scienceqa/llava_scienceqa_test3.jsonl \
        --image-folder /data/ouyangxc/data/science/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/caption_sqa.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/scienceqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python llava_uhd/eval/eval_science_qa.py \
    --base-dir /data/ouyangxc/data/science \
    --result-file $output_file \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
