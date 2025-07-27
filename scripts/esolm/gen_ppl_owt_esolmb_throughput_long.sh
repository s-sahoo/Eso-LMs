#!/bin/bash

T=1000000
alpha_eval=1.0

sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
  --alpha_0 1.0 \
  --T "${T}" \
  --batch_size 256 \
  --num_batches 6 \
  --ckpt_path "/mnt/weka/home/zhihan.yang/checkpoints/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt" \
  --profile_throughput True \
  --length 2048 \
  --samples_path "/mnt/weka/home/zhihan.yang/Eso-LMs/log/throughput/esolmb_2048/eval_${alpha_eval}/${T}.json"

sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
  --alpha_0 1.0 \
  --T "${T}" \
  --batch_size 64 \
  --num_batches 6 \
  --ckpt_path "/mnt/weka/home/zhihan.yang/checkpoints/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt" \
  --profile_throughput True \
  --length 8192 \
  --samples_path "/mnt/weka/home/zhihan.yang/Eso-LMs/log/throughput/esolmb_8192/eval_${alpha_eval}/${T}.json"

sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
  --alpha_0 1.0 \
  --T "${T}" \
  --batch_size 32 \
  --num_batches 6 \
  --ckpt_path "/mnt/weka/home/zhihan.yang/checkpoints/owt-esolmb-alpha0-1-313000/checkpoints/14-250000.ckpt" \
  --profile_throughput True \
  --length 16384 \
  --samples_path "/mnt/weka/home/zhihan.yang/Eso-LMs/log/throughput/esolmb_16384/eval_${alpha_eval}/${T}.json"
