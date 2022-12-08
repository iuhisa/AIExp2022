import torch
import os

print('GPU 使用可能か')
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print('環境変数 CUDA_VISIBLE_DEVICES')
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    
    print('使用できるGPUの数')
    print(torch.cuda.device_count())

    print('デフォルトGPU番号')
    print(torch.cuda.current_device())
