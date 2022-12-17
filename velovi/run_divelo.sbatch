#!/bin/bash

#SBATCH -o /home/icb/%u/workspace/stdout/divelo_%j.job
#SBATCH -e /home/icb/%u/code/velo_interpret/velovi/output/run/divelo_err_%j.job
#SBATCH -J divelo
#SBATCH -c 4 #CPU cores required
#SBATCH --mem=20G #Memory required
#SBATCH -t 4:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1 #How many GPUs you request, use 1

source $HOME/.bashrc
conda activate thesis

python  $HOME/code/velo_interpret/velovi/run_divelo.py $HOME/code/velo_interpret/velovi/data/Pancreas/pancreas_data_annotations.h5ad data/Pancreas/pancreas_data_hp_sweep.h5ad
