#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train.py --A_dataroot photo_ukiyoe --B_dataroot ukiyoe --load_size 158 --crop_size 128 --save_epoch_interval 10