#!/bin/bash

HF_HUB_OFFLINE=True
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="llava-uhd-v1.5-7b-lora"
CKPT="${MODEL_NAME}_1"
SPLIT="llava_vizwiz"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava_uhd.eval.model_vqa_loader  \
        --model-path ./checkpoints/$MODEL_NAME \
        --model-base /data/llm_common/llava-v1.5-7b \
        --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
        --image-folder /data/ouyangxc/data/vizwiz/test \
        --answers-file ./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vizwiz/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$MODEL_NAME.json
