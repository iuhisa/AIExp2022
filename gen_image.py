from package.model import get_model
from package.util import visualize
from package.data import get_dataloader
from package.options.test_options import TestOptions
from PIL import Image
import os.path as osp
from package.util import check_dir
from tqdm import tqdm
import numpy as np
import torch
import download_dataset as dd

if __name__ == '__main__':
    opt = TestOptions().parse()
    save_dir = osp.join('datasets', opt.dst_dataroot, 'images')
    check_dir(save_dir)
    opt.batch_size = 1
    opt.no_flip = True
    opt.A_datatype = 'isolated_w_path'
    opt.preprocess = 'resize'
    dataloader = get_dataloader(opt, domain='A')

    print('num of test images = %d' % (len(dataloader)))

    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    model.eval()
    for data in tqdm(dataloader):
        in_data = {'A':data['img']}
        model.set_input(in_data)
        model.test()
        gen_data = model.get_fake()
        gen_data = gen_data[0].detach().cpu().numpy()
        gen_data = ((gen_data*0.5 + 0.5)*255).clip(0,255)
        gen_img = Image.fromarray(gen_data.astype(np.uint8).transpose(1,2,0))

        # print(data['path'])

        # gen_img = gen_img.transpose(1,2,0)
        in_path = data['path']
        file_name = osp.basename(data['path'][0])
        gen_img.save(osp.join(save_dir, file_name))
        
    dd.make_img_list(osp.join('datasets', opt.dst_dataroot), train=0, test=1, val=0)
