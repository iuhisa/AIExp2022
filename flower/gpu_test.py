import torch

print('GPU 使用可能か')
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print('使用できるGPUの数')
    print(torch.cuda.device_count())

    print('デフォルトGPU番号')
    print(torch.cuda.current_device())
