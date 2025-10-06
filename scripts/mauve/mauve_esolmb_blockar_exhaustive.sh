#!/bin/bash

export BASE_LOG_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log"

echo $BASE_LOG_DIR

# ============================  MAIN LOOP  ==========================
SUBCONTEXT_LENS=(16 32 64 128 256)

for SUBCONTEXT_LEN in "${SUBCONTEXT_LENS[@]}"; do
  sbatch scripts/mauve/mauve.sh \
    --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
    --samples_path "${BASE_LOG_DIR}/samples_5120/esolmb_blockar/train_1/eval_1/${SUBCONTEXT_LEN}.json" \
    --log_dir "${BASE_LOG_DIR}/mauve/esolmb_blockar/train_1/eval_1/T_${SUBCONTEXT_LEN}"
done