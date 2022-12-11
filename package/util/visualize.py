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

    # def save(model, dataloaders, device, save_path):
    #     # if generate image using CycleGAN
    #     fig = self.draw_figure_CycleGAN(model, *dataloaders, device)
    #     fig.savefig(save_path)

    # def show(model, dataloaders, device, save_path):
    #     # if generate image using CycleGAN
    #     fig = self.draw_figure_CycleGAN(model, *dataloaders, device)
    #     plt.show(fig)


    def draw_figure_CycleGAN(model, src_dataloader, dst_dataloader, device):
        # src_image_path_list = glob.glob(osp.join('demo_data', 'src', '*'))
        # dst_image_path_list = glob.glob(osp.join('demo_data', 'dst', '*'))
        # transform = FlowerTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # src_images = []
        # dst_images = []
        # for path in src_image_path_list:
        #     image = Image.open(path)
        #     image = transform(image).to(device)
        #     src_images.append(image)
        # for path in dst_image_path_list:
        #     image = Image.open(path)
        #     image = transform(image).to(device)
        #     dst_images.append(image)
        src = next(iter(src_dataloader))
        dst = next(iter(dst_dataloader))
        src = src.to(device)
        dst = dst.to(device)
        
        regen_images = model(torch.stack(src))
        n = len(dst)
        fig = plt.figure(num=1, clear=True, tight_layout=True) # memory leak 対策
        axes = fig.subplots(3, n)
        for i in range(0, n):
            s = src[i].detach().cpu().numpy()
            d = dst[i].detach().cpu().numpy()
            r = regen_images[i].detach().cpu().numpy()
            s = ((s*0.5 + 0.5)*255).clip(0, 255)
            d = ((d*0.5 + 0.5)*255).clip(0, 255)
            r = ((r*0.5 + 0.5)*255).clip(0, 255)

            axes[0, i].set_xticks([])
            axes[1, i].set_xticks([])
            axes[2, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[1, i].set_yticks([])
            axes[2, i].set_yticks([])
            axes[0, i].imshow(s.astype(np.uint8).transpose(1,2,0))
            axes[1, i].imshow(d.astype(np.uint8).transpose(1,2,0))
            axes[2, i].imshow(r.astype(np.uint8).transpose(1,2,0))
        return fig