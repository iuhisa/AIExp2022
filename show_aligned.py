import argparse
import cv2
import os.path as osp
import download_dataset as dd
from package.data.dataset import get_filepath_list
import numpy as np
# import cv2
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', type=str, nargs='*')
    parser.add_argument('-s', '--size', type=int)
    args = parser.parse_args()

    img_template_paths = [osp.join('datasets', dataset_path, 'images' , '%s.jpg') for dataset_path in args.datasets]
    txtfile_path = osp.join('datasets', args.datasets[0], 'test.txt')

    for line in open(txtfile_path):
        file_id = line.strip()
        imgs = []
        concatted_img = None
        for i in range(len(args.datasets)):
            img_path = img_template_paths[i] % file_id
            img = Image.open(img_path)
            img = img.resize((args.size, args.size))
            imgs.append(np.array(img))
            if concatted_img is None:
                concatted_img = Image.new('RGB', (img.width * len(args.datasets), img.height))
            concatted_img.paste(img, (i*img.width, 0))


        fig = plt.figure(figsize=(10,10))
        # print(imgs[0].shape)
        # disp_img = np.concatenate(imgs, axis=0)
        # print(disp_img.shape)
        # plt.imshow(Image.fromarray(disp_img.transpose(1,2,0)))
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.imshow(concatted_img)
        plt.show()
        # cv2.waitKey(0)

    # list_imagepaths = [get_filepath_list(a, phase='test', list_len_max=float('inf')) for a in args.datasets]
    # imagepaths_len = np.array([len(a) for a in list_imagepaths])
    # min_i = np.argmin(imagepaths_len)

    # imagepaths = list_imagepaths[min_i]

    # for 

    # print(imagepaths_len)