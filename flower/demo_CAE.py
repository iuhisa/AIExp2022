import torch
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from model import CAE
from preprocessing import FlowerTransform

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
    fig = plt.figure(figsize=(12, 9))
    for i in range(0, n):
        s = src_images[i].detach().cpu().numpy()
        d = dst_images[i].detach().cpu().numpy()
        r = regen_images[i].detach().cpu().numpy()
        s = ((s*0.5 + 0.5)*255).clip(0, 255)
        d = ((d*0.5 + 0.5)*255).clip(0, 255)
        r = ((r*0.5 + 0.5)*255).clip(0, 255)
        plt.subplot(3, n, i+1)
        plt.imshow(s.astype(np.uint8).transpose(1,2,0))

        plt.subplot(3, n, n+i+1)
        plt.imshow(d.astype(np.uint8).transpose(1,2,0))

        plt.subplot(3, n, n*2+i+1)
        plt.imshow(r.astype(np.uint8).transpose(1,2,0))

    plt.savefig(out_path)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoEncoder = CAE()
    autoEncoder.to(device)
    autoEncoder.load_state_dict(torch.load(osp.join('weight', 'CAE_final.th'), map_location=device))
    autoEncoder.eval()
    demo(autoEncoder, device)