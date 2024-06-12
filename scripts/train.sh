#!/bin/bash
export HF_HUB_OFFLINE=True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# export https_proxy="http://10.3.73.27:7891"
export WANDB_MODE=dryrun
export WANDB_PROJECT=UHD
# export WANDB_ENTITY=964730078

# 记录开始时间
start_time=$(date +%s)

deepspeed --master_port 29642 llava_uhd/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data2/llm_common/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data2/datasets/llava/blip_laion_cc_sbu_558k.json \
    --image_folder /data2/datasets/llava/llava1.5_pretrain/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

# 记录结束时间
end_time=$(date +%s)
# 计算总耗时（秒）
total_time=$((end_time - start_time))
# 转换为小时和分钟
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
# 输出耗时
echo "Total time: $hours hours $minutes mins"

# 记录开始时间
start_time=$(date +%s)

deepspeed --master_port 29642 llava_uhd/train.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --image_folder /data2/datasets/llava/ft_datasets \
    --data_path /data2/datasets/llava/llava_v1_5_mix665k.json \
    --model_name_or_path /data2/llm_common/vicuna-7b-v1.5 \
    --version v1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-uhd-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

# 记录结束时间
end_time=$(date +%s)
# 计算总耗时（秒）
total_time=$((end_time - start_time))
# 转换为小时和分钟
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
# 输出耗时
echo "Total time: $hours hours $minutes mins"