'''
Small Cycle GAN Model
'''
from .base_model import BaseModel
from . import networks

class TestCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['G_A']
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        # AtoB = (self.opt.direction == 'AtoB')
        self.real_A = input['A'].to(self.device)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)

    def get_fake(self):
        return self.fake_B

    def optimize(self):
        pass