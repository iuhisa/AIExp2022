import torch
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image

from package.model.networks import CAE
from package.data.transform import FlowerTransform

import time
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from package.model import get_model
from package.util import visualize
from package.data import get_unpair_dataloader
from package.options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.batch_size = opt.save_image_num
    opt.no_flip = True

    A_dataloader, B_dataloader = get_unpair_dataloader(opt)
    print('num of test images = %d, %d' % (len(A_dataloader), len(B_dataloader)))

    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    if opt.eval:
        model.eval()
    A_data = next(iter(A_dataloader))
    B_data = next(iter(B_dataloader))
    data = {'A':A_data, 'B':B_data}
    model.set_input(data)
    model.test()

    save_n = opt.save_image_num
    visualizer.show_imgs(model.real_A[:save_n], model.fake_B[:save_n], model.rec_A[:save_n], id='AtoB')
    visualizer.show_imgs(model.real_B[:save_n], model.fake_A[:save_n], model.rec_B[:save_n], id='BtoA')

    
'''legacy
def demo(autoEncoder, device, out_path='demo.png'):
    src_image_path_list = glob.glob(osp.join('demo_data', 'src', '*'))
    dst_image_path_list = glob.glob(osp.join('demo_data', 'dst', '*'))
    transform = FlowerTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    src_images = []
    dst_images = []
    for path in src_image_path_list:
        image = Image.open(path)
        image = transform(image).to(device)
        src_images.append(image)
    for path in dst_image_path_list:
        image = Image.open(path)
        image = transform(image).to(device)
        dst_images.append(image)
    
    regen_images = autoEncoder(torch.stack(src_images))
    n = len(dst_images)
    fig = plt.figure(num=1, clear=True, tight_layout=True) # memory leak 対策
    axes = fig.subplots(3, n)
    for i in range(0, n):
        s = src_images[i].detach().cpu().numpy()
        d = dst_images[i].detach().cpu().numpy()
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

    fig.savefig(out_path)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoEncoder = CAE()
    autoEncoder.to(device)
    autoEncoder.load_state_dict(torch.load(osp.join('weight', '12-08_00-26-57', 'CAE_0.th'), map_location=device))
    autoEncoder.eval()
    demo(autoEncoder, device)
'''