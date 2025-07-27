#!/usr/bin/env bash

T=8

CKPT_PATH="/mnt/weka/home/zhihan.yang/checkpoints/owt-mdlm-302860/checkpoints/14-250000.ckpt"
SAMPLES_BASE_DIR="/mnt/weka/home/zhihan.yang/Eso-LMs/log/throughput"

sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
  --T "$T" \
  --seed 1 \
  --batch_size 256 \
  --num_batches 6 \
  --ckpt_path $CKPT_PATH \
  --profile_throughput True \
  --samples_path "${SAMPLES_BASE_DIR}/mdlm_2048/${T}.json" \
  --length 2048

sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
  --T "$T" \
  --seed 1 \
  --batch_size 64 \
  --num_batches 6 \
  --ckpt_path $CKPT_PATH \
  --profile_throughput True \
  --samples_path "${SAMPLES_BASE_DIR}/mdlm_8192/${T}.json" \
  --length 8192

sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
  --T "$T" \
  --seed 1 \
  --batch_size 32 \
  --num_batches 6 \
  --ckpt_path $CKPT_PATH \
  --profile_throughput True \
  --samples_path "${SAMPLES_BASE_DIR}/mdlm_16384/${T}.json" \
  --length 16384
