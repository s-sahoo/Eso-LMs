#!/bin/bash
#SBATCH -J eval_owt_ar
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) data="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

DATA_DIR=${HOME}/data/esolm
# CKPT_PATH=${HOME}/checkpoints/owt-ar-323034/checkpoints/14-250000.ckpt
CKPT_PATH=${HOME}/checkpoints/owt-ar-prepend-mask-431810/checkpoints/14-250000.ckpt

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=ppl_eval \
  loader.batch_size=4 \
  loader.eval_batch_size=4 \
  data=${data} \
  data.insert_valid_eos=False \
  +data.insert_valid_special=False \
  algo=ar \
  algo.prepend_token=mask \
  model.length=1024 \
  data.cache_dir=${DATA_DIR} \
  eval.checkpoint_path=$CKPT_PATH \
  eval.generate_samples=False \
  +wandb.offline=true
