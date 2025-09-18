#!/bin/bash

# -------------------------------------------------------------------
# 1. Map **training-time** alpha_0  →  checkpoint path
#    (edit keys & paths to match your checkpoints)
declare -A CKPT_PATHS=(
  [0.125]="/mnt/weka/home/zhihan.yang/checkpoints/owt-esolmb-alpha0-0d125-L10240-finetune-584401/checkpoints/0-1000.ckpt"
)

# -------------------------------------------------------------------
# 2. Evaluation-time alpha_0 sweep
EVAL_ALPHAS=(0.125)

# ============================  MAIN LOOP  ==========================
for alpha_train in "${!CKPT_PATHS[@]}"; do
  ckpt_path="${CKPT_PATHS[$alpha_train]}"

  for alpha_eval in "${EVAL_ALPHAS[@]}"; do
    
    Ts=(1000000)

    for T in "${Ts[@]}"; do

      mkdir -p log/samples/train_${alpha_train}/eval_${alpha_eval}

      sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
        --alpha_0 "${alpha_eval}" \
        --T "${T}" \
        --batch_size 1 \
        --num_batches 1 \
        --ckpt_path "${ckpt_path}" \
        --length 10240 \
        --profile_throughput False \
        --samples_path "/mnt/weka/home/zhihan.yang/Eso-LMs/log/samples_1000_L10240/esolmb/train_${alpha_train}/eval_${alpha_eval}/${T}.json"

    done
  done
done
