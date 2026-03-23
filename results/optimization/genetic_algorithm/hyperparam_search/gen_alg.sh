#!/bin/bash
#SBATCH --job-name=gen_alg
#SBATCH --partition=qany
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --error=job-%A_%a.err
#SBATCH --output=job-%A_%a.out
#SBATCH --array=1-9

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"
python3 ../run_genetic_algorithm.py `awk "NR == $SLURM_ARRAY_TASK_ID" script_params.txt`

echo "========= Job Finished  at `date` =========="
