#!/usr/bin/env bash

sbatch scripts/mdlm/gen_ppl_owt_mdlm.sh \
    --T 1 \
    --batch_size 32 \
    --num_batches 4 \
    --ckpt_path /mnt/weka/home/zhihan.yang/checkpoints/owt-mdlm-382896/checkpoints/14-250000.ckpt \
    --profile_throughput False \
    --length 1024 \
    --seed 1 \
    --use_block_ar_order True \
    --samples_path /mnt/weka/home/zhihan.yang/Eso-LMs/log/samples_5120/mdlm_use_block_ar_order/samples.json
