#!/bin/bash
#SBATCH -J eval_esolm                  # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --alpha_0) alpha_0="$2"; shift ;;
        --batch_split) batch_split="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo $alpha_0
echo $batch_split
echo $ckpt_path

nvidia-smi
nvcc --version

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=ppl_eval \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  +data.insert_valid_special=False \
  model=small \
  model.length=1024 \
  algo=esolm \
  algo.alpha_0=$alpha_0 \
  algo.batch_split=$batch_split \
  algo.diffusion_attn_mode=causal \
  algo.diffusion_shuffle=True \
  algo.sequential_attn_mode=causal  \
  algo.sequential_shuffle=True \
  eval.checkpoint_path=$ckpt_path \
  sampling.num_sample_batches=0 \
  +wandb.offline=true
