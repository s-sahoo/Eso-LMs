#!/bin/bash
#SBATCH -J sample_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

nvidia-smi
nvcc --version

# checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/mdlm.ckpt
checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diff-clean-s-owt-no-t-mQ4fQG-param-subs_data-openwebtext-split/checkpoints/15-250000.ckpt

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  loader.batch_size=1 \
  loader.eval_batch_size=1 \
  data=openwebtext-split \
  model=small \
  algo=mdlm \
  sampling.predictor=ancestral_cache \
  eval.checkpoint_path=${checkpoint_path} \
  sampling.num_sample_batches=8 \
  sampling.p_nucleus=1.0 \
  +wandb.offline=true
