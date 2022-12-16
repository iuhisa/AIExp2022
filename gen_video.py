import time
import torch
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

from package.model import get_model
from package.util import visualize
from package.data import get_dataloader
from package.options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.no_flip = True
    opt.batch_size = 1
    direction = opt.direction
    dataloader = get_dataloader(opt, domain='A')
    dataloader = iter(dataloader)
    datatype = opt.A_datatype


    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    save_dir = osp.join(opt.checkpoints_dir, opt.name)
    FPS = 10.0
    out_filepath = opt.A_dataroot + '.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    out_video = None

    if opt.eval:
        model.eval()

    for A_data in tqdm(dataloader):
        if out_video == None:
            _, _, h, w = A_data.size()
            out_video = cv2.VideoWriter(osp.join(save_dir,out_filepath), codec, FPS, (2*w, h))
        data = {'A':A_data}
        model.set_input(data)
        if datatype == 'isolated':
            in_data = A_data
        elif datatype == 'sequential':
            in_data = A_data[:, 2]
        in_img = in_data.detach().cpu().numpy()[0]
        in_img = ((in_img*0.5 + 0.5)*255).clip(0,255).astype(np.uint8)

        model.test()

        fake_data = model.get_fake() # [batch_size, channel_size, height, width]
        gen_data = fake_data.detach().cpu().numpy()[0]

        out_img = gen_data
        out_img = ((out_img*0.5 + 0.5)*255).clip(0,255).astype(np.uint8)

        min1 = min(in_img.shape[1], out_img.shape[1])# 入力縦横のサイズ奇数の時、1だけずれることがある。
        min2 = min(in_img.shape[2], out_img.shape[2])
        in_img = in_img[:, :min1, :min2]
        out_img = out_img[:, :min1, :min2]

        cat_img = np.concatenate([in_img, out_img], axis=2)
        cat_img = cat_img.transpose(1,2,0)[:,:,[2,1,0]]
        out_video.write(cat_img)
    
    out_video.release()
