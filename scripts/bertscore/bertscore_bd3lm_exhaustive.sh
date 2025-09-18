#!/bin/bash

export BASE_LOG_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log"
export BASE_LOG_DIR_BD3LM="/mnt/weka/home/zhihan.yang/bd3lms/logs"

echo $BASE_LOG_DIR

BLOCKSIZE=4
Ts=(1 2 4 8 16)

for T in "${Ts[@]}"; do

  sbatch scripts/mauve/mauve.sh \
    --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
    --samples_path "${BASE_LOG_DIR_BD3LM}/samples_5120/samples_bd3lm_blocksize${BLOCKSIZE}_T${T}_seed1.json" \
    --log_dir "${BASE_LOG_DIR}/mauve/bd3lm/blocksize_${BLOCKSIZE}/T_${T}"

done

BLOCKSIZE=8
Ts=(2 4 8 16 32)

for T in "${Ts[@]}"; do

  sbatch scripts/mauve/mauve.sh \
    --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
    --samples_path "${BASE_LOG_DIR_BD3LM}/samples_5120/samples_bd3lm_blocksize${BLOCKSIZE}_T${T}_seed1.json" \
    --log_dir "${BASE_LOG_DIR}/mauve/bd3lm/blocksize_${BLOCKSIZE}/T_${T}"

done

BLOCKSIZE=16
Ts=(4 8 16 32 64)

for T in "${Ts[@]}"; do

  sbatch scripts/mauve/mauve.sh \
    --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
    --samples_path "${BASE_LOG_DIR_BD3LM}/samples_5120/samples_bd3lm_blocksize${BLOCKSIZE}_T${T}_seed1.json" \
    --log_dir "${BASE_LOG_DIR}/mauve/bd3lm/blocksize_${BLOCKSIZE}/T_${T}"

done