#!/bin/bash

export BASE_LOG_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log"

echo $BASE_LOG_DIR

Ts=(8 16 32 48 64 128 256 1024 4096)

for T in "${Ts[@]}"; do

  sbatch scripts/bertscore/bertscore.sh \
    --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
    --samples_path "${BASE_LOG_DIR}/samples_5120/mdlm/${T}.json" \
    --log_dir "${BASE_LOG_DIR}/bertscore/mdlm/T_${T}"

done
