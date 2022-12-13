#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 gen_video.py --A_dataroot nature_video_test --B_dataroot hoge --direction AtoB --load_size 256 --crop_size 256 --name 12-13_17-02-36 --model cycle_gan --batch_size 1 --num_threads 1 --epoch latest --netG resnet_9blocks