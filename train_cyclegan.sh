#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:4

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train.py --A_dataroot fake_mask --B_dataroot real_mask2 --load_size 286 --crop_size 256 --save_epoch_interval 10 --model cycle_gan --batch_size 16 --batch_multiplier 4 --netG resnet_9blocks --num_threads 4 --pool_size 50