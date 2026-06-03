#!/bin/bash
#SBATCH --job-name=act_learn
#SBATCH --partition=qgpu
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --error=genetic_alg_dataset_no_saas/model_3_active_finale/job.err
#SBATCH --output=genetic_alg_dataset_no_saas/model_3_active_finale/job.out
#SBATCH --gres=gpu:1

python3 ../active_learning.py -model_ckpt=../../diffusion_model_parameters/8000k_sample_models_100_no_saas/trained_models/model_3/checkpoints/best_epoch=epoch=605-val=val_loss=1.4584.ckpt -init_traj=../../diffusion_model_parameters/datasets_100/genetic_algorithm_8000_no_saas.traj -pre_sample_db=genetic_alg_dataset_no_saas/model_3_active_finale/pre_estim.db -n_loops=1 -n_samples_per_loop=1000 -m_index=100 -proj_name=active_learning_no_saas -out=genetic_alg_dataset_no_saas/model_3_active_finale -dev=cuda 