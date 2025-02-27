'''
get_model でmodelを作成
modelは、networkをもち、networkに対してforward,backward,optimize等を統括して指示する。
'''
import torch.nn as nn
import importlib

from .cycle_gan_model import CycleGANModel
from .recycle_gan_model import RecycleGANModel
from .test_cycle_gan_model import TestCycleGANModel
from .test_recycle_gan_model import TestRecycleGANModel
from .pix2pix_model import Pix2PixModel
from .test_pix2pix_model import TestPix2PixModel
from .base_model import BaseModel


def get_model(opt):
    if opt.model == 'cycle_gan':
        model = CycleGANModel(opt)
    elif opt.model == 'recycle_gan':
        model = RecycleGANModel(opt)
    elif opt.model == 'test_cycle_gan':
        model = TestCycleGANModel(opt)
    elif opt.model == 'test_recycle_gan':
        model = TestRecycleGANModel(opt)
    elif opt.model == 'pix2pix':
        model = Pix2PixModel(opt)
    elif opt.model == 'test_pix2pix':
        model = TestPix2PixModel(opt)

    # else model_name == 'CycleGAN':
    # ...

    return model


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "package.model." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options