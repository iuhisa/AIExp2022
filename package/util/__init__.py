'''
雑多な関数を定義
'''
import glob
import os.path as osp
import os
import torch.cuda as cuda
import cv2
import numpy as np


def make_filepath_list(root = 'dataset'):
    train_dirpath = osp.join(root, 'train')
    test_dirpath = osp.join(root, 'test')

    train_dst_filepath_list = glob.glob(osp.join(train_dirpath, 'dst', '*.jpg'))
    train_src_filepath_list = glob.glob(osp.join(train_dirpath, 'src', '*.jpg'))
    test_dst_filepath_list = glob.glob(osp.join(test_dirpath, 'dst', '*.jpg'))
    test_src_filepath_list = glob.glob(osp.join(test_dirpath, 'src', '*.jpg'))
    
    return train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list



# ディレクトリがなければ作る。再帰的に作れる。
def check_dir(path):
    os.makedirs(path, exist_ok=True)

def get_gpu_list():
    if cuda.is_available():
        ret = list(range(cuda.current_device(), cuda.current_device() + cuda.device_count()))
        print(f'GPU: {cuda.get_device_name()}, count: {cuda.device_count()}')
    else:
        ret = []
        print('CPU')
    return ret


def clip_image(img: np.ndarray, h: int, w: int, type: str = 'center') -> np.ndarray:
    h_, w_ = img.shape[:2]
    top = bottom = left = right = 0
    if h_ < h or w_ < w:
        return None
    if type == 'center':
        if h_%2 == 0 and h%2 == 0:
            top = h_/2-h/2
            bottom = h_/2+h/2-1
        elif h_%2 == 0 and h%2 != 0:
            top = h_/2-int(h/2)-1
            bottom = h_/2+int(h/2)-1
        elif h_%2 != 0 and h%2 == 0:
            top = int(h_/2)-h/2
            bottom = int(h_/2)+h/2-1
        else:
            top = int(h_/2)-int(h/2)
            bottom = int(h_/2)+int(h/2)

        if w_%2 == 0 and w%2 == 0:
            left = w_/2-w/2
            right = w_/2+w/2-1
        elif w_%2 == 0 and w%2 != 0:
            left = w_/2-int(w/2)-1
            right = w_/2+int(w/2)-1
        elif w_%2 != 0 and w%2 == 0:
            left = int(w_/2)-w/2
            right = int(w_/2)+w/2-1
        else:
            left = int(w_/2)-int(w/2)
            right = int(w_/2)+int(w/2)
    return img[int(top): int(bottom)+1, int(left): int(right)+1]

def get_stats(opt, domain):
    if domain == 'A':
        statsfile_path = osp.join('datasets', opt.A_dataroot, opt.phase+'_stats.txt')
    elif domain == 'B':
        statsfile_path = osp.join('datasets', opt.B_dataroot, opt.phase+'_stats.txt')

    if not osp.exists(statsfile_path):
        print(f'stats file does not exist: {statsfile_path}')
        return {'mean':(0.5, 0.5, 0.5), 'std':(0.5, 0.5, 0.5)}

    with open(statsfile_path, 'r') as f:
        line = f.readline()
        r_mean, r_std, g_mean, g_std, b_mean, b_std = map(float, line.split(','))
    return {'mean':(r_mean, g_mean, b_mean), 'std':(r_std, g_std, b_std)}
