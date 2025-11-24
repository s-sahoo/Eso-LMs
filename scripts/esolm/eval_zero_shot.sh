#!/bin/bash
#SBATCH -J zeroshot
#SBATCH --partition=main
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

export HYDRA_FULL_ERROR=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo) algo="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
        --data) data="$2"; shift ;;
        --alpha_0) alpha_0="$2"; shift ;;
        --alpha_0_eval) alpha_0_eval="$2"; shift ;;
        --batch_split) batch_split="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo $algo
echo $ckpt_path
echo $data
echo $alpha_0
echo $alpha_0_eval

DATA_DIR=${HOME}/data/esolm

nvidia-smi
nvcc --version

srun python -u -m main \
  mode=ppl_eval \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=${data} \
  data.cache_dir=${DATA_DIR} \
  data.insert_valid_eos=False \
  +data.insert_valid_special=True \
  +data.insert_train_special=True \
  eval.generate_samples=False \
  model=small \
  algo=${algo} \
  algo.batch_split=${batch_split} \
  algo.alpha_0=${alpha_0_eval} \
  algo.diffusion_shuffle=True \
  algo.diffusion_attn_mode=causal \
  algo.sequential_shuffle=True \
  algo.sequential_attn_mode=causal \
  model.length=1024 \
  eval.checkpoint_path=$ckpt_path \
  +wandb.offline=true  > $PWD/log/${algo}_${alpha_0}_${alpha_0_eval}_${data}_no_valid_eos.log
