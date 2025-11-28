#!/bin/bash
#SBATCH --job-name=bayes_opt
#SBATCH --partition=q48
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

python3 ../../run_optimization.py -rnd_seed=42 -m_index=100 -n_batches=6 -batch_size=50 -out=results/timings_mem-per-cpu/5G -opt=bayesian_opt -model=WWL-GPR -time=True 