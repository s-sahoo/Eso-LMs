#!/bin/bash
#SBATCH -J train_sudoku_esolm            # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=128000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=thickstun         # Request partition
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
RUN_NAME=sudoku-esolmb-alpha0-1-${SLURM_JOB_ID}
CHECKPOINT_DIR=/share/thickstun/zhihan/checkpoints/esolm/${RUN_NAME}
export HF_HOME=/share/thickstun/zhihan/.cache/huggingface

srun python -u -m main \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  model=tiny \
  data=sudoku \
  wandb.name=${RUN_NAME} \
  algo=esolm \
  algo.alpha_0=1.0 \
  algo.batch_split=1.0 \
  algo.diffusion_shuffle=True \
  algo.diffusion_attn_mode=causal \
  algo.keep_masks_unshuffled=True \
  algo.loss_type=elbo \
  model.length=81 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  trainer.val_check_interval=3525 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=35250 \
  trainer.log_every_n_steps=100 \
  trainer.max_steps=1000000 \
  data.cache_dir=${DATA_DIR} \
  hydra.run.dir=${CHECKPOINT_DIR}
