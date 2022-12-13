import time
import torch
import cv2

from package.model import get_model
from package.util import visualize
from package.data import get_dataloader
from package.options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.no_flip = True
    direction = opt.direction
    AtoB = True if direction == 'AtoB' else False
    if AtoB:
        dataloader = get_dataloader(opt, domain='A')
        datatype = opt.A_datatype
    else:
        dataloader = get_dataloader(opt, domain='B')
        datatype = opt.B_datatype


    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    FPS = 10.0
    filepath = 'output.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    w = opt.crop_size
    h = opt.crop_size
    video = cv2.VideoWriter(filepath, codec, FPS, (w, h))

    if opt.eval:
        model.eval()

    for _data in dataloader:
        if AtoB:
            A_data = _data
            B_data = torch.zeros_like(A_data)
        else:
            B_data = _data
            A_data = torch.zeros_like(B_data)
        data = {'A':A_data, 'B':B_data}
        model.set_input(data)
        model.forward()

        fake_data = model.get_fake() # [batch_size, channel_size, height, width]
        if AtoB:
            gen_data = fake_data['B'].detach().cpu().numpy()
        else:
            gen_data = fake_data['A'].detach().cpu().numpy()

        for i in range(gen_data.shape[0]):
            img = gen_data[i]
            video.write(img)
    
    video.release()
