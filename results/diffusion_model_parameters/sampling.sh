#!/bin/bash
#SBATCH --job-name=diff_sample
#SBATCH --partition=qgpu
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --error=job-%A_%a.err
#SBATCH --output=job-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --array=1-10

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"
python3 benchmark_sampling.py `awk "NR == $SLURM_ARRAY_TASK_ID" script_params_sample.txt`

echo "========= Job Finished  at `date` =========="
