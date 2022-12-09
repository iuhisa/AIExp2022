'''
活性化関数
'''
import torch
import torch.nn as nn

class TanhExp(nn.Module):
    def __init__(self):
        super(TanhExp, self).__init__()
    def forward(self, x):
        return x*torch.tanh(torch.exp(x))