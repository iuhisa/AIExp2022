# '''
# Copyed Recycle GAN Model
# '''
# import torch
# import itertools
# from .base_model import BaseModel
# from . import networks
# from ..util.image_pool import ImagePool

# class RecycleGANModel(BaseModel):
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         parser.set_defaults(no_dropout=True)
#         if is_train:
#             # netP == 'prediction'は今は無し
#             parser.add_argument('--netP', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128 | prediction]')
#             parser.add_argument('--npf', type=int, default=64, help='# of pred filters in the last conv layer')
#             parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#             parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
#             parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
#             parser.add_argument('--adversarial_loss_p', action='store_true', help='also train the prediction model with an adversarial loss')
#         return parser
    
#     def __init__(self, opt):
#         BaseModel.__init__(self, opt)
#         self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'pred_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'pred_B']
#         if self.isTrain:
#             self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'P_A', 'P_B']
#         else:
#             self.model_names = ['G_A', 'G_B', 'P_A', 'P_B']
#         self.adversarial_loss_p = opt.adversarial_loss_p
#         self.visual_names = ['real_A0', 'real_A1', 'real_A2',
#                              'real_B0', 'real_B1', 'real_B2',
#                              'fake_A0', 'fake_A1', 'fake_A2',
#                              'fake_B0', 'fake_B1', 'fake_B2',
#                              'rec_A'  , 'rec_B',
#                              'pred_A2', 'pred_B2']
    
#         self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#         if opt.netP == 'prediction':
#             self.netP_A = networks.define_G(opt.input_nc, opt.input_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.netP_B = networks.define_G(opt.output_nc, opt.output_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         else:
#             self.netP_A = networks.define_G(2 * opt.input_nc, opt.input_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.netP_B = networks.define_G(2 * opt.output_nc, opt.output_nc, opt.npf, opt.netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#         if self.isTrain:
#             self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

#             if opt.lambda_identity > 0.0:
#                 assert(opt.input_nc == opt.output_nc)
#             self.fake_A_pool = ImagePool(opt.pool_size)
#             self.fake_B_pool = ImagePool(opt.pool_size)

#             self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
#             self.criterionCycle = torch.nn.L1Loss()
#             self.criterionIdt = torch.nn.L1Loss()

#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netP_A.parameters(), self.netP_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D_A)
#             self.optimizers.append(self.optimizer_D_B)

#     def set_input(self, input):
#         # AtoB = (self.opt.direction == 'AtoB')
#         self.real_A0 = input['A'][:, 0].to(self.device)
#         self.real_A1 = input['A'][:, 1].to(self.device)
#         self.real_A2 = input['A'][:, 2].to(self.device)
#         self.real_B0 = input['B'][:, 0].to(self.device)
#         self.real_B1 = input['B'][:, 1].to(self.device)
#         self.real_B2 = input['B'][:, 2].to(self.device)

#     def forward(self):
#         pass
#         # self.fake_B0, 1
#         # self.fake_A0, 1 が必要？

#         # self.fake_B = self.netG_A(self.real_A)
#         # self.rec_A = self.netG_B(self.fake_B)
#         # self.fake_A = self.netG_B(self.real_B)
#         # self.rec_B = self.netG_A(self.fake_A)
    
#     def test(self):
#         self.fake_B0 = self.netG_A(self.real_A0)
#         self.fake_B1 = self.netG_A(self.real_A1)
#         if self.opt.netP == 'prediction':
#             self.fake_B2 = self.netP_B(self.fake_B0, self.fake_B1)
#         else:
#             self.fake_B2 = self.netP_B(torch.cat((self.fake_B0, self.fake_B1), 1))
#         self.rec_A = self.netG_B(self.fake_B2)

#         self.fake_A0 = self.netG_B(self.real_B0)
#         self.fake_A1 = self.netG_B(self.real_B1)

#         if self.opt.netP == 'prediction':
#             self.fake_A2 = self.netP_A(self.fake_A0, self.fake_A1)
#         else:
#             self.fake_A2 = self.netP_A(torch.cat((self.fake_A0, self.fake_A1), 1))
#         self.rec_B = self.netG_A(self.fake_A2)

#         if self.opt.netP == 'prediction':
#             self.pred_A2 = self.netP_A(self.real_A0, self.real_A1)
#             self.pred_B2 = self.netP_B(self.real_B0, self.real_B1)
#         else:
#             self.pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1), 1))
#             self.pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1), 1))


#     def backward_D_basic(self, netD, real, fake):
#         pred_real = netD(real)
#         loss_D_real = self.criterionGAN(pred_real, True)
#         pred_fake = netD(fake.detach())
#         loss_D_fake = self.criterionGAN(pred_fake, False)
#         loss_D = (loss_D_real + loss_D_fake) * 0.5
#         loss_D.backward()
#         return loss_D

#     def backward_D_A(self):
#         fake_B0 = self.fake_B_pool.query(self.fake_B0)
#         loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_B0, fake_B0)
#         fake_B1 = self.fake_B_pool.query(self.fake_B1)
#         loss_D_A1 = self.backward_D_basic(self.netD_A, self.real_B1, fake_B1)
#         fake_B2 = self.fake_B_pool.query(self.fake_B2)
#         loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_B2, fake_B2)
#         pred_B = self.fake_B_pool.query(self.pred_B2)
#         loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_B2, pred_B)

#         self.loss_D_A = loss_D_A0.item() + loss_D_A1.item() + loss_D_A2.item() + loss_D_A3.item()
    
#     def backward_D_B(self):
#         fake_A0 = self.fake_A_pool.query(self.fake_A0)
#         loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_A0, fake_A0)
#         fake_A1 = self.fake_A_pool.query(self.fake_A1)
#         loss_D_B1 = self.backward_D_basic(self.netD_B, self.real_A1, fake_A1)
#         fake_A2 = self.fake_A_pool.query(self.fake_A2)
#         loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A2, fake_A2)
#         pred_A = self.fake_A_pool.query(self.pred_A2)
#         loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_A2, pred_A)

#         self.loss_D_B = loss_D_B0.item() + loss_D_B1.item() + loss_D_B2.item() + loss_D_B3.item()
    
#     def backward_G(self):
#         lambda_idt = self.opt.lambda_identity
#         lambda_A = self.opt.lambda_A
#         lambda_B = self.opt.lambda_B

#         # identity loss は無し？
#         if lambda_idt > 0:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0
#             # self.idt_A = self.netG_A(self.real_B)
#             # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
#             # self.idt_B = self.netG_B(self.real_A)
#             # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
#         else:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0

#         # Loss GAN  A -> B
#         self.fake_B0 = self.netG_A(self.real_A0)
#         loss_G_A0 = self.criterionGAN(self.netD_A(self.fake_B0), True)
#         self.fake_B1 = self.netG_A(self.real_A1)
#         loss_G_A1 = self.criterionGAN(self.netD_A(self.fake_B1), True)
        
#         if self.opt.netP == 'prediction':
#             self.fake_B2 = self.netP_B(self.fake_B0, self.fake_B1)
#         else:
#             self.fake_B2 = self.netP_B(torch.cat((self.fake_B0, self.fake_B1), 1))
#         loss_G_A2 = self.criterionGAN(self.netD_A(self.fake_B2), True)

#         # Loss GAN  B -> A
#         self.fake_A0 = self.netG_B(self.real_B0)
#         loss_G_B0 = self.criterionGAN(self.netD_B(self.fake_A0), True)
#         self.fake_A1 = self.netG_B(self.real_B1)
#         loss_G_B1 = self.criterionGAN(self.netD_B(self.fake_A1), True)
        
#         if self.opt.netP == 'prediction':
#             self.fake_A2 = self.netP_A(self.fake_A0, self.fake_A1)
#         else:
#             self.fake_A2 = self.netP_A(torch.cat((self.fake_A0, self.fake_A1), 1))
#         loss_G_B2 = self.criterionGAN(self.netD_B(self.fake_A2), True)

#         # Loss pred
#         if self.opt.netP == 'prediction':
#             self.pred_A2 = self.netP_A(self.real_A0, self.real_A1)
#         else:
#             self.pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1), 1))
#         loss_pred_A = self.criterionCycle(self.pred_A2, self.real_A2) * lambda_A
#         if self.opt.netP == 'prediction':
#             self.pred_B2 = self.netP_B(self.real_B0, self.real_B1)
#         else:
#             self.pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1), 1))
#         loss_pred_B = self.criterionCycle(self.pred_B2, self.real_B2) * lambda_B

#         if self.adversarial_loss_p:
#             pred_fake = self.netD_B(self.pred_A2)
#             loss_pred_A_adversarial = self.criterionGAN(pred_fake, True)
#             pred_fake = self.netD_A(self.pred_B2)
#             loss_pred_B_adversarial = self.criterionGAN(pred_fake, True)
#         else:
#             loss_pred_A_adversarial = 0
#             loss_pred_B_adversarial = 0

#         # Loss Cycle(Recycle)
#         self.rec_A = self.netG_B(self.fake_B2)
#         loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A2) * lambda_A

#         self.rec_B = self.netG_A(self.fake_A2)
#         loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B2) * lambda_B

#         loss_G = (loss_G_A0 + loss_G_A1 + loss_G_A2 + loss_G_B0 + loss_G_B1 + loss_G_B2 + loss_cycle_A + loss_cycle_B
#          + loss_pred_A + loss_pred_B + self.loss_idt_A + self.loss_idt_B + loss_pred_A_adversarial + loss_pred_B_adversarial)

#         loss_G.backward()

#         # 保存
#         self.loss_G_A = loss_G_A0 + loss_G_A1 + loss_G_A2
#         self.loss_G_B = loss_G_B0 + loss_G_B1 + loss_G_B2
#         self.loss_cycle_A = loss_cycle_A
#         self.loss_cycle_B = loss_cycle_B
#         self.loss_pred_A = loss_pred_A
#         self.loss_pred_B = loss_pred_B

#     def optimize(self):
#         self.forward()

#         self.set_requires_grad([self.netD_A, self.netD_B], False)
#         self.optimizer_G.zero_grad()
#         self.backward_G()
#         self.optimizer_G.step()

#         self.set_requires_grad([self.netD_A, self.netD_B], True)
#         self.optimizer_D_A.zero_grad()
#         self.backward_D_A()
#         self.optimizer_D_A.step()

#         self.optimizer_D_B.zero_grad()
#         self.backward_D_B()
#         self.optimizer_D_B.step()
