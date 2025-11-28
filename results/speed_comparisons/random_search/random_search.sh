#!/bin/bash
#SBATCH --job-name=rnd_search
#SBATCH --partition=q48
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.err
#SBATCH --output=job.out

python3 ../../run_optimization.py -rnd_seed=42 -m_index=100 -n_batches=7 -batch_size=100 -out=results/timings_partition/q48 -opt=random_search -model=WWL-GPR 
