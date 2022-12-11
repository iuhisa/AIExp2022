import torch
import os.path as osp
from abc import ABC, abstractmethod
from collections import OrderedDict

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = osp.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda' if len(self.gpu_ids) > 0 else 'cpu')
        self.loss_names = []
        self.model_names = []
        self.optimizers = []

        torch.backends.cudnn.benchmark = True

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def optimize(self):
        pass


    def setup(self, opt):
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks()

    def eval(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.eval()
    def test(self):
        with torch.no_grad():
            self.forward()
    
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = osp.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            if len(self.gpu_ids) > 0:
                torch.save(net.module.cpu().state_dict(), save_path)
                net.to(self.device)
            else:
                torch.save(net.cpu().state_dict(), save_path)
    
    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = osp.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)
    
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad