'''
画像表示用の関数たち
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp
from collections import OrderedDict
import pandas as pd
from ..model import CycleGANModel, RecycleGANModel

class Visualizer():
    def __init__(self, opt, loss_names):
        self.opt = opt
        self.save_dir = osp.join(opt.checkpoints_dir, opt.name)
        self.logs = []
        self.epoch_losses = {}
        self.total_losses = {}
        for name in loss_names:
            self.total_losses[name] = []
            self.epoch_losses[name] = 0

    def store_loss(self, losses: OrderedDict):
        for k in losses.keys():
            self.epoch_losses[k] += losses[k]

    def save_loss(self, epoch):
        for k in self.total_losses.keys():
            self.total_losses[k].append(self.epoch_losses[k])
        log_epoch = dict(**{'epoch': epoch}, **self.epoch_losses)
        self.logs.append(log_epoch)
        df = pd.DataFrame(self.logs)
        df.to_csv(osp.join(self.save_dir, 'log.csv'), index=False)
        for k in self.epoch_losses.keys():
            self.epoch_losses[k] = 0

    def plot_loss(self):
        fig = plt.figure(num=1, clear=True) # memory leak 対策
        ax = fig.subplots()
        n = len(self.logs)
        x = [i for i in range(1, n+1)]
        for k in self.total_losses.keys():
            ax.plot(x, self.total_losses[k], label='loss_'+k)
        ax.set_xlabel('epoch')
        ax.legend()
        fig.savefig(osp.join(self.save_dir, 'loss_plot.pdf'))

    def save_images(self, model, epoch=''):
        if isinstance(model, CycleGANModel):
            save_n = self.opt.save_image_num
            visuals = model.get_current_visuals()
            fig = self.make_fig(torch.stack([visuals['real_A'][:save_n], visuals['fake_B'][:save_n], visuals['rec_A'][:save_n]]))
            fig.savefig(osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, 'AtoB')))
            fig = self.make_fig([visuals['real_B'][:save_n], visuals['fake_A'][:save_n], visuals['rec_B'][:save_n]])
            fig.savefig(osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, 'BtoA')))
            
        elif isinstance(model, RecycleGANModel):
            visuals = model.get_current_visuals()
            fig = self.make_fig(torch.stack([
                torch.stack([visuals['real_A0'][0], visuals['real_A1'][0], visuals['real_A2'][0]]),
                torch.stack([visuals['fake_B0'][0], visuals['fake_B1'][0], visuals['fake_B2'][0]])
            ]))
            fig.savefig(osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, 'AtoB')))
            fig = self.make_fig(torch.stack([
                torch.stack([visuals['real_B0'][0], visuals['real_B1'][0], visuals['real_B2'][0]]),
                torch.stack([visuals['fake_A0'][0], visuals['fake_A1'][0], visuals['fake_A2'][0]])
            ]))
            fig.savefig(osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, 'BtoA')))

    
    # def save_imgs(self, imgs:torch.Tensor, epoch='', id=''):
    #     '''
    #     parameters
    #     ------------
    #         imgs.shape: torch.size([kinds of image, num of image per 1 kind, channels, height, width])
    #     '''
    #     fig = self.make_fig(imgs)
    #     save_path = osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, id))
    #     fig.savefig(save_path)


    # def save_imgs(self, real_imgs:torch.Tensor, fake_imgs:torch.Tensor, rec_imgs:torch.Tensor, epoch='', id=''):
    #     assert((real_imgs.dim() == fake_imgs.dim()) and (fake_imgs.dim() == rec_imgs.dim()))

    #     fig = self.make_fig(torch.stack([real_imgs, fake_imgs, rec_imgs]))
    #     save_path = osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, id))
    #     fig.savefig(save_path)

    def make_fig(self, imgs: torch.Tensor):
        '''
        parameters
        ------------
            imgs.shape: torch.size([kinds of image, num of image per 1 kind, channels, height, width])
        '''
        #A to B to A
        #B to A to B
        
        m = imgs.size()[0]
        n = imgs.size()[1]
        assert(m > 1 or n > 1)
        fig = plt.figure(num=1, clear=True, tight_layout=True) # memory leak 対策
        axes = fig.subplots(m, n)
        for i in range(0, m):
            for j in range(0, n):
                img = imgs[i][j].detach().cpu().numpy()
                img = ((img*0.5 + 0.5)*255).clip(0,255)
                axes[i, j].set_xticks([])
                axes[i, j].imshow(img.astype(np.uint8).transpose(1,2,0))
        return fig

    def show_imgs(self, real_imgs:torch.Tensor, fake_imgs:torch.Tensor, rec_imgs:torch.Tensor, id=''):
        assert((real_imgs.dim() == fake_imgs.dim()) and (fake_imgs.dim() == rec_imgs.dim()))

        fig = self.make_fig(torch.cat([real_imgs, fake_imgs, rec_imgs]))
        plt.title(id)
        plt.show()
