'''
各タスク用のtransformsたち
'''
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp
from ..util import get_stats
import random
import numpy as np

# def get_transform(opt, grayscale=False, convert=True, method=transforms.InterpolationMode.BICUBIC):
def get_transform(opt, domain, grayscale=False, params=None, convert=True, method=Image.BICUBIC): # for ist cluster
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.cop_size, method)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if convert:
        transform_list.append(transforms.ToTensor())
        if grayscale:
            transform_list.append(transforms.Normalize((0.5,),(0.5,)))
        else:
            s = get_stats(opt, domain)
            transform_list.append(transforms.Normalize(s['mean'],s['std']))

    return transforms.Compose(transform_list)
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}
class InvNormalize:
    def __init__(self, mean, std):
        self.invTrans = transforms.Compose(
        [
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [1/std[0], 1/std[1], 1/std[2]]),
            transforms.Normalize(mean = [-mean[0], -mean[1], -mean[2]], std = [ 1., 1., 1. ]),
        ])
    def __call__(self, x):
        return self.invTrans(x)


class FlowerTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.data_transform(img)

# def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
def __scale_width(img, target_size, crop_size, method=Image.BICUBIC): # for ist cluster
    # method = __transforms2pil_resize(method) # dont use in ist cluster
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

# IST Cluster は torchvision のバージョンが古すぎるのでこの関数がいらない。
def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True