#!/bin/bash
#SBATCH -J bertscore
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
        --human_references_path) human_references_path="$2"; shift ;;
        --samples_path) samples_path="$2"; shift ;;
        --log_dir) log_dir="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo $human_references_path
echo $samples_path

nvidia-smi
nvcc --version

export HYDRA_FULL_ERROR=1

srun python -u -m bertscore \
  --human-references-path=$human_references_path \
  --samples-path=$samples_path \
  --log-dir=$log_dir
