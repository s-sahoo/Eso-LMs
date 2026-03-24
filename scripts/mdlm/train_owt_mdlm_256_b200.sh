#!/bin/bash
#SBATCH -J train_owt_mdlm            # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=genai-thickstun-highpri             # Request partition
#SBATCH --constraint="[b200]"
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2                  # Type/number of GPUs needed
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

nvidia-smi
nvcc --version

DATA_DIR=/share/thickstun/zhihan/data/blockdiffusion
RUN_NAME=owt-mdlm-${SLURM_JOB_ID}
CHECKPOINT_DIR=/share/thickstun/zhihan/checkpoints/esolm/${RUN_NAME}
export HF_HOME=/share/thickstun/zhihan/.cache/huggingface

srun python -u -m main \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  model=small \
  data=openwebtext-split \
  +data.insert_train_special=False \
  +data.insert_valid_special=False \
  wandb.name=${RUN_NAME} \
  algo=mdlm \
  model.length=256 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  trainer.val_check_interval=10000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
  trainer.log_every_n_steps=1000 \
  trainer.max_steps=1000000 \
  data.cache_dir=${DATA_DIR} \
  hydra.run.dir=${CHECKPOINT_DIR}
