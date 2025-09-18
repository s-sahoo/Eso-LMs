#!/usr/bin/env bash

# Ts=(8 16 32 48 64 128 256 1024 4096)
Ts=(20 24)

CKPT_PATH="/mnt/weka/home/zhihan.yang/checkpoints/owt-mdlm-302860/checkpoints/14-250000.ckpt"
SAMPLES_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log/samples_5120/mdlm"

for T in "${Ts[@]}"; do
  if (( T >= 1024 )); then  # smaller batch size for large T (nfe < T for single batch)
    for seed in 1 2 3 4; do
      sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
        --T "$T" \
        --seed "$seed" \
        --batch_size 1 \
        --num_batches 1280 \
        --ckpt_path $CKPT_PATH \
        --profile_throughput False \
        --samples_path "${SAMPLES_DIR}/${T}_seed${seed}.json"
    done
  else  # larger batch size for small T (nfe ~ T)
    sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
      --T "$T" \
      --seed 1 \
      --batch_size 32 \
      --num_batches 160 \
      --ckpt_path $CKPT_PATH \
      --profile_throughput False \
      --samples_path "${SAMPLES_DIR}/${T}.json"
  fi
done
