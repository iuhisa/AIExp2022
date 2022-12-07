#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
#SBATCH -n 1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_CAE.py