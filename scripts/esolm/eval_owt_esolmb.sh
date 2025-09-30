#!/bin/bash
#SBATCH -J eval_owt_esolm
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --alpha_0) alpha_0="$2"; shift ;;
        --batch_split) batch_split="$2"; shift ;;
        --num_iw_orders) num_iw_orders="$2"; shift ;;
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

DATA_DIR=${HOME}/data/esolm

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
  eval.num_iw_orders=$num_iw_orders \
  sampling.num_sample_batches=0 \
  data.cache_dir=${DATA_DIR} \
  +wandb.offline=true
