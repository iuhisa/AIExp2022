'''
Small Recycle GAN Model
'''
import torch
from .base_model import BaseModel
from . import networks

class TestRecycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--netP', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128 | prediction]')
            parser.add_argument('--npf', type=int, default=64, help='# of pred filters in the last conv layer')
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.model_names = ['G_A', 'P_B']
        self.visual_names = ['real_A0', 'real_A1', 'real_A2',
                             'fake_B']
    
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.netP == 'prediction':
            self.netP_B = networks.define_G(opt.output_nc, opt.output_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netP_B = networks.define_G(2 * opt.output_nc, opt.output_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        # AtoB = (self.opt.direction == 'AtoB')
        self.real_A0 = input['A'][:, 0].to(self.device)
        self.real_A1 = input['A'][:, 1].to(self.device)
        self.real_A2 = input['A'][:, 2].to(self.device)

    def forward(self):
        pass
        # self.fake_B0, 1
        # self.fake_A0, 1 が必要？

        # self.fake_B = self.netG_A(self.real_A)
        # self.rec_A = self.netG_B(self.fake_B)
        # self.fake_A = self.netG_B(self.real_B)
        # self.rec_B = self.netG_A(self.fake_A)
    
    def test(self):
        fake_B0 = self.netG_A(self.real_A0)
        fake_B1 = self.netG_A(self.real_A1)
        if self.opt.netP == 'prediction':
            fake_B2 = self.netP_B(fake_B0, fake_B1)
        else:
            fake_B2 = self.netP_B(torch.cat((fake_B0, fake_B1), 1))
        
        self.fake_B = (self.netG_A(self.real_A2) + fake_B2) * 0.5

    def get_fake(self):
        return self.fake_B

    def optimize(self):
        pass