#!/bin/bash

# -------------------------------------------------------------------
# 1. Map **training-time** alpha_0  →  checkpoint path
#    (edit keys & paths to match your checkpoints)
declare -A CKPT_PATHS=(
  [1]="${HOME}/checkpoints/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt"
  [0.5]="${HOME}/checkpoints/owt-esolmb-alpha0-0d5-313003/checkpoints/14-250000.ckpt"
  [0.25]="${HOME}/checkpoints/owt-esolmb-alpha0-0d25-313002/checkpoints/14-250000.ckpt"
  [0.125]="${HOME}/checkpoints/owt-esolmb-alpha0-0d125-313001/checkpoints/14-250000.ckpt"
)

# -------------------------------------------------------------------
# 2. Evaluation-time alpha_0 sweep
EVAL_ALPHAS=(1 0.5 0.25 0.0625 0)

# ============================  MAIN LOOP  ==========================
for alpha_train in "${!CKPT_PATHS[@]}"; do
  ckpt_path="${CKPT_PATHS[$alpha_train]}"

  for alpha_eval in "${EVAL_ALPHAS[@]}"; do
    # Choose T-set based on alpha_eval
    if [[ $(printf "%.4f" "${alpha_eval}") == "1.0000" ]]; then
      Ts=(16 32 64 128 256 1024 4096)
    elif [[ $(printf "%.4f" "${alpha_eval}") == "0.0000" ]]; then
      Ts=(1)  # doesn't matter in this case since it's like ar
    else
      Ts=(16 128 256 1024)
    fi

    for T in "${Ts[@]}"; do

      sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
        --alpha_0 "${alpha_eval}" \
        --T "${T}" \
        --batch_size 256 \
        --num_batches 20 \
        --ckpt_path "${ckpt_path}" \
        --profile_throughput False \
        --samples_path ${HOME}/Eso-LMs/log/samples_5120/esolmb/train_${alpha_train}/eval_${alpha_eval}/${T}.json

    done
  done
done
