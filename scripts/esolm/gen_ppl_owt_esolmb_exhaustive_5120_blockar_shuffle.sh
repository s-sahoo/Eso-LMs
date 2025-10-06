#!/bin/bash

# -------------------------------------------------------------------
# 1. Map **training-time** alpha_0  →  checkpoint path
#    (edit keys & paths to match your checkpoints)
declare -A CKPT_PATHS=(
  [1]="${HOME}/checkpoints/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt"
)

# -------------------------------------------------------------------
# 2. Evaluation-time alpha_0 sweep
EVAL_ALPHAS=(1)

# ============================  MAIN LOOP  ==========================
for alpha_train in "${!CKPT_PATHS[@]}"; do
  for alpha_eval in "${EVAL_ALPHAS[@]}"; do

    if [[ $(printf "%.4f" "${alpha_eval}") == "1.0000" ]]; then
      SUBCONTEXT_LENS=(16 32 64 128 256 512 1024)
    fi

  ckpt_path="${CKPT_PATHS[$alpha_train]}"
  
    for SUBCONTEXT_LEN in "${SUBCONTEXT_LENS[@]}"; do

      sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
        --alpha_0 "${alpha_eval}" \
        --T "${T}" \
        --batch_size 512 \
        --num_batches 10 \
        --ckpt_path "${ckpt_path}" \
        --profile_throughput False \
        --subcontext_len "${SUBCONTEXT_LEN}" \
        --subcontext_shuffle True \
        --samples_path ${HOME}/Eso-LMs/log/samples_5120/esolmb_blockar_shuffle/train_${alpha_train}/eval_${alpha_eval}/${SUBCONTEXT_LEN}.json
    
    done
  done
done