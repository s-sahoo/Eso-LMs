#!/usr/bin/env bash

CKPT_PATH=${HOME}/checkpoints/owt-mdlm-382896/checkpoints/14-250000.ckpt
SAMPLES_PATH=${HOME}/Eso-LMs/log/samples_5120/mdlm_use_ar_order/samples.json

sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
    --T 1 \
    --batch_size 32 \
    --num_batches 160 \
    --ckpt_path $CKPT_PATH \
    --profile_throughput False \
    --length 1024 \
    --seed 1 \
    --use_ar_order True \
    --samples_path $SAMPLES_PATH
