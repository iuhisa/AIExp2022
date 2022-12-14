#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:4

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train.py --A_dataroot nature_video --B_dataroot ukiyoe_video --A_datatype sequential --B_datatype sequential --load_size 286 --crop_size 256 --save_epoch_interval 10 --model recycle_gan --batch_size 16 --netG resnet_6blocks --netP unet_256 --pool_size 50 --max_dataset_size 5000