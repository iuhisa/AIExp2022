import torch
import os

print('GPU 使用可能か')
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print('使用できるGPUの数')
#SBATCH --gres=gpu:2 とすると、torch.cuda.device_count() == 2になる。
    print(torch.cuda.device_count())
    print('デフォルトGPU番号')
    print(torch.cuda.current_device())
    print('環境変数 CUDA_VISIBLE_DEVICES')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    print(type(os.environ['CUDA_VISIBLE_DEVICES']))
