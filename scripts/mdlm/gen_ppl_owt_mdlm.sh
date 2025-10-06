#!/bin/bash
#SBATCH -J sample_owt_mdlm
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
        --T) T="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_batches) num_batches="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
        --profile_throughput) profile_throughput="$2"; shift ;;
        --length) length="$2"; shift ;;  # optional
        --samples_path) samples_path="$2"; shift ;;  # optional
        --seed) seed="$2"; shift ;;  # optional
        --use_ar_order) use_ar_order="$2"; shift ;;  # optional
        --use_block_ar_order) use_block_ar_order="$2"; shift ;;  # optional
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo $T
echo $batch_size
echo $num_batches
echo $ckpt_path
echo $profile_throughput
echo $samples_path
echo $use_ar_order

nvidia-smi
nvcc --version

export HYDRA_FULL_ERROR=1

srun python -u -m main \
  mode=sample_eval \
  loader.eval_batch_size=$batch_size \
  model=small \
  ${length:+model.length="$length"} \
  algo=mdlm \
  eval.checkpoint_path=$ckpt_path \
  sampling.num_sample_batches=$num_batches \
  sampling.p_nucleus=0.9 \
  sampling.steps=$T \
  sampling.profile_throughput=$profile_throughput \
  ${samples_path:+eval.generated_samples_path="$samples_path"} \
  ${seed:+seed=$seed} \
  ${use_ar_order:+sampling.use_ar_order=$use_ar_order} \
  ${use_block_ar_order:+sampling.use_block_ar_order=$use_block_ar_order} \
  +wandb.offline=true
