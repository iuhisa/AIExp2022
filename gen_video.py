import time
import torch
import cv2
import numpy as np

from package.model import get_model
from package.util import visualize
from package.data import get_dataloader
from package.options.test_options import TestOptions

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TestOptions().parse()
    opt.no_flip = True
    opt.batch_size = 1
    direction = opt.direction
    dataloader = get_dataloader(opt, domain='A')
    datatype = opt.A_datatype
    # AtoB = True if direction == 'AtoB' else False
    # if AtoB:
    #     dataloader = get_dataloader(opt, domain='A')
    #     datatype = opt.A_datatype
    # else:
    #     dataloader = get_dataloader(opt, domain='B')
    #     datatype = opt.B_datatype


    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    FPS = 10.0
    in_filepath = 'input.mp4'
    out_filepath = 'output.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    w = opt.crop_size
    h = opt.crop_size
    in_video = cv2.VideoWriter(in_filepath, codec, FPS, (w, h))
    out_video = cv2.VideoWriter(out_filepath, codec, FPS, (w, h))

    if opt.eval:
        model.eval()

    for A_data in dataloader:
        # if AtoB:
        #     A_data = _data
        #     B_data = torch.zeros_like(A_data)
        # else:
        #     B_data = _data
        #     A_data = torch.zeros_like(B_data)
        data = {'A':A_data}
        model.set_input(data)
        if datatype == 'isolated':
            in_data = A_data
        elif datatype == 'sequential':
            in_data = A_data[:, 2]
        # for i in range(in_data.size()[0]):
        in_img = in_data[0].detach().cpu().numpy()
        in_img = ((in_img*0.5 + 0.5)*255).clip(0,255).astype(np.uint8)
        in_video.write(in_img.transpose(1,2,0)[:,:,[2,1,0]])

        # model.forward()
        model.test()

        fake_data = model.get_fake() # [batch_size, channel_size, height, width]
        # if AtoB:
            # original_data = A_data.detach().cpu().numpy()
        gen_data = fake_data[0].detach().cpu().numpy()
        # else:
            # original_data = B_data.detach().cpu().numpy()
            # gen_data = fake_data['A'].detach().cpu().numpy()

        # for i in range(gen_data.shape[0]):
        out_img = gen_data
        out_img = ((out_img*0.5 + 0.5)*255).clip(0,255).astype(np.uint8)
        out_video.write(out_img.transpose(1,2,0)[:,:,[2,1,0]])
    
    in_video.release()
    out_video.release()
