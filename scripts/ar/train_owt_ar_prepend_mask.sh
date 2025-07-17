#!/bin/bash
#SBATCH -J train_owt_ar
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

nvidia-smi
nvcc --version

DATA_DIR=${HOME}/data/esolm
RUN_NAME=owt-ar-prepend-mask-${SLURM_JOB_ID}
CHECKPOINT_DIR=${HOME}/checkpoints/${RUN_NAME}

srun python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  model=small \
  data=openwebtext-split \
  +data.insert_train_special=False \
  +data.insert_valid_special=False \
  wandb.name=${RUN_NAME} \
  algo=ar \
  algo.prepend_token=mask \
  model.length=1024 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  trainer.val_check_interval=10000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
  trainer.log_every_n_steps=1000 \
  trainer.max_steps=1000000 \
  data.cache_dir=${DATA_DIR} \
  hydra.run.dir=${CHECKPOINT_DIR}
