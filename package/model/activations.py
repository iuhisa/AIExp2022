'''
活性化関数
'''
import torch
import torch.nn as nn

class TanhExp(nn.Module):
    '''
        x -> x * tanh ( exp ( x ) )
    '''
    def __init__(self, *args):
        super(TanhExp, self).__init__()
    def forward(self, x):
        return x*torch.tanh(torch.exp(x))