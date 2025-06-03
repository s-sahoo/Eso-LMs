#!/bin/bash
#SBATCH -J sample_esolm                # Job name
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

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --alpha_0) alpha_0="$2"; shift ;;
        --T) T="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_batches) num_batches="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo $alpha_0
echo $T
echo $batch_size
echo $num_batches
echo $ckpt_path

nvidia-smi
nvcc --version

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  loader.eval_batch_size=$batch_size \
  model.length=1024 \
  model=small \
  algo=esolm \
  algo.alpha_0=$alpha_0 \
  algo.diffusion_attn_mode=causal \
  algo.diffusion_shuffle=True \
  algo.sequential_attn_mode=causal  \
  algo.sequential_shuffle=True \
  eval.checkpoint_path=$ckpt_path \
  sampling.kv_cache=True \
  sampling.num_sample_batches=$num_batches \
  sampling.steps=$T \
  sampling.p_nucleus=0.9 \
  +wandb.offline=true
