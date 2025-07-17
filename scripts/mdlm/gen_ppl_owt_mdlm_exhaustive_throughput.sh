#!/usr/bin/env bash

Ts=(8 16 32 64 128 256 1024 4096)

CKPT_PATH="/mnt/weka/home/zhihan.yang/checkpoints/owt-mdlm-302860/checkpoints/14-250000.ckpt"
SAMPLES_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log/throughput/mdlm"

for T in "${Ts[@]}"; do
  sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
    --T "$T" \
    --seed 1 \
    --batch_size 512 \
    --num_batches 6 \
    --ckpt_path $CKPT_PATH \
    --profile_throughput True \
    --samples_path "${SAMPLES_DIR}/${T}.json"
done
