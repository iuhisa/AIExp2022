'''
get_model でmodelを作成
modelは、networkをもち、networkに対してforward,backward,optimize等を統括して指示する。
'''
import torch.nn as nn

from .cae_model import CAEModel


def get_model(model_name, config):
    # if model_name == 'CAE':
    model = CAEModel()

    # else model_name == 'CycleGAN':
    # ...

    return model


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose2d') != -1 or classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)