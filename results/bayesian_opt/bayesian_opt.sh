#!/bin/bash
#SBATCH --job-name=bayes_opt
#SBATCH --partition=q48
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out
#SBATCH --array=1-10

echo "========= Job started  at `date` ==========" >> job.out

echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"
python3 ../run_optimization.py `awk "NR == $SLURM_ARRAY_TASK_ID" script_params.txt`

echo "========= Job Finished  at `date` ==========" >> job.out
