#!/bin/bash
#SBATCH -J sample_owt_ar
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

nvidia-smi
nvcc --version

checkpoint_path="/mnt/weka/home/zhihan.yang/checkpoints/owt-ar-323034/checkpoints/14-250000.ckpt"

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  loader.eval_batch_size=512 \
  model.length=1024 \
  model=small \
  algo=ar \
  eval.checkpoint_path=$checkpoint_path \
  sampling.kv_cache=True \
  sampling.num_sample_batches=6 \
  sampling.p_nucleus=0.9 \
  sampling.profile_throughput=True \
  +wandb.offline=true
