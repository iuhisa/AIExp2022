'''
画像表示用の関数たち
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
import os.path as osp
from collections import OrderedDict
import pandas as pd

class Visualizer():
    def __init__(self, opt, loss_names):
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
        for k in self.epoch_losses.keys():
            self.total_losses[k].append(self.epoch_losses[k])
        log_epoch = {'epoch': epoch} + self.epoch_losses
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
        for k in self.losses.keys():
            ax.plot(x, self.losses[k], label='loss_'+k)
        ax.set_xlabel('epoch')
        ax.legend()
        fig.savefig(osp.join(self.save_dir, 'loss_plot.pdf'))
    
    def save_imgs(self, imgs:torch.Tensor, epoch='', id=''):
        '''
        parameters
        ------------
            imgs.shape: torch.size([kinds of image, num of image per 1 kind, channels, height, width])
        '''
        fig = self.make_fig(imgs)
        save_path = osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, id))
        fig.savefig(save_path)


    def save_imgs(self, real_imgs:torch.Tensor, fake_imgs:torch.Tensor, rec_imgs:torch.Tensor, epoch='', id=''):
        assert((real_imgs.dim() == fake_imgs.dim()) and (fake_imgs.dim() == rec_imgs.dim()))

        fig = self.make_fig(torch.cat([real_imgs, fake_imgs, rec_imgs]))
        save_path = osp.join(self.save_dir, '{}_{}_images.jpg'.format(epoch, id))
        fig.savefig(save_path)

    def make_fig(self, imgs: torch.Tensor):
        '''
        parameters
        ------------
            imgs.shape: torch.size([kinds of image, num of image per 1 kind, channels, height, width])
        '''
        #A to B to A
        #B to A to B
        m = imgs.dim()
        n = imgs[0].dim()
        fig = plt.figure(num=1, clear=True, tight_layout=True) # memory leak 対策
        axes = fig.subplots(m, n)
        for i in range(0, m):
            for j in range(0, n):
                img = imgs[i][j].detach().cpu().numpy()
                img = ((img*0.5 + 0.5)*255).clip(0,255)
                axes[i, j].set_xticks([])
                axes[i, j].imshow(img.astype(np.uint8).transpose(1,2,0))
            # real = real_imgs[i].detach().cpu().numpy()
            # fake = fake_imgs[i].detach().cpu().numpy()
            # rec = rec_imgs[i].detach().cpu().numpy()
            # s = ((s*0.5 + 0.5)*255).clip(0, 255)
            # d = ((d*0.5 + 0.5)*255).clip(0, 255)
            # r = ((r*0.5 + 0.5)*255).clip(0, 255)

            # axes[0, i].set_xticks([])
            # axes[1, i].set_xticks([])
            # axes[2, i].set_xticks([])
            # axes[0, i].set_yticks([])
            # axes[1, i].set_yticks([])
            # axes[2, i].set_yticks([])
            # axes[0, i].imshow(s.astype(np.uint8).transpose(1,2,0))
            # axes[1, i].imshow(d.astype(np.uint8).transpose(1,2,0))
            # axes[2, i].imshow(r.astype(np.uint8).transpose(1,2,0))
        return fig
    def show_imgs(self, real_imgs:torch.Tensor, fake_imgs:torch.Tensor, rec_imgs:torch.Tensor, id=''):
        assert((real_imgs.dim() == fake_imgs.dim()) and (fake_imgs.dim() == rec_imgs.dim()))

        fig = self.make_fig(torch.cat([real_imgs, fake_imgs, rec_imgs]))
        plt.title(id)
        plt.show()        


    # def save(model, dataloaders, device, save_path):
    #     # if generate image using CycleGAN
    #     fig = self.draw_figure_CycleGAN(model, *dataloaders, device)
    #     fig.savefig(save_path)

    # def show(model, dataloaders, device, save_path):
    #     # if generate image using CycleGAN
    #     fig = self.draw_figure_CycleGAN(model, *dataloaders, device)
    #     plt.show(fig)


    # def draw_figure_CycleGAN(model, src_dataloader, dst_dataloader, device):
    #     src = next(iter(src_dataloader))
    #     dst = next(iter(dst_dataloader))
    #     src = src.to(device)
    #     dst = dst.to(device)
        
    #     regen_images = model(torch.stack(src))
    #     n = len(dst)
    #     fig = plt.figure(num=1, clear=True, tight_layout=True) # memory leak 対策
    #     axes = fig.subplots(3, n)
    #     for i in range(0, n):
    #         s = src[i].detach().cpu().numpy()
    #         d = dst[i].detach().cpu().numpy()
    #         r = regen_images[i].detach().cpu().numpy()
    #         s = ((s*0.5 + 0.5)*255).clip(0, 255)
    #         d = ((d*0.5 + 0.5)*255).clip(0, 255)
    #         r = ((r*0.5 + 0.5)*255).clip(0, 255)

    #         axes[0, i].set_xticks([])
    #         axes[1, i].set_xticks([])
    #         axes[2, i].set_xticks([])
    #         axes[0, i].set_yticks([])
    #         axes[1, i].set_yticks([])
    #         axes[2, i].set_yticks([])
    #         axes[0, i].imshow(s.astype(np.uint8).transpose(1,2,0))
    #         axes[1, i].imshow(d.astype(np.uint8).transpose(1,2,0))
    #         axes[2, i].imshow(r.astype(np.uint8).transpose(1,2,0))
    #     return fig