#!/bin/bash
#SBATCH --job-name=rnd_search
#SBATCH --partition=qany
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --error=job-%A_%a.err
#SBATCH --output=job-%A_%a.out
#SBATCH --array=1-10

echo "========= Job started  at `date` =========="

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"
python3 run_random_search.py `awk "NR == $SLURM_ARRAY_TASK_ID" script_params.txt`

echo "========= Job Finished  at `date` =========="
