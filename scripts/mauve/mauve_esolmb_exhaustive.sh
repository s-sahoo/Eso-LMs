#!/bin/bash

TRAIN_ALPHAS=(1 0.5 0.25 0.125)
# EVAL_ALPHAS=(1 0.5 0.25 0.0625)
EVAL_ALPHAS=(0)

export BASE_LOG_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log"

echo $BASE_LOG_DIR

# ============================  MAIN LOOP  ==========================
for alpha_train in "${TRAIN_ALPHAS[@]}"; do
  for alpha_eval in "${EVAL_ALPHAS[@]}"; do
    if [[ $(printf "%.4f" "${alpha_eval}") == "1.0000" ]]; then
      Ts=(16 32 64 128 256 1024 4096)
    elif [[ $(printf "%.4f" "${alpha_eval}") == "0.0000" ]]; then
      Ts=(1)
    else
      Ts=(16 128 1024)
    fi

    for T in "${Ts[@]}"; do

      sbatch scripts/mauve/mauve.sh \
        --human_references_path "${BASE_LOG_DIR}/human_references/human_references_no_special_tokens.json" \
        --samples_path "${BASE_LOG_DIR}/samples_5120/esolmb/train_${alpha_train}/eval_${alpha_eval}/${T}.json" \
        --log_dir "${BASE_LOG_DIR}/mauve/esolmb/train_${alpha_train}/eval_${alpha_eval}/T_${T}"

    done
  done
done
