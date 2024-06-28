#!/bin/bash

# 设置环境变量 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 执行需要执行的脚本
bash scripts/eval/mme.sh
bash scripts/eval/gqa.sh
bash scripts/eval/pope.sh
bash scripts/eval/sqa.sh
bash scripts/eval/textvqa.sh
bash scripts/eval/vizwiz.sh
bash scripts/eval/vqav2.sh
# bash scripts/eval/okvqa.sh
# bash scripts/eval/mmbench.sh
# bash scripts/eval/mmbench_cn.sh