#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 gen_video.py --A_dataroot mountain_video --B_dataroot hoge --direction AtoB --load_size 512 --crop_size 512 --name 12-14_17-02-36 --model cycle_gan --batch_size 1 --epoch latest --netG resnet_9blocks