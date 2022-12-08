#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:2
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 gpu_test.py