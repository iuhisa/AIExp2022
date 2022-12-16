'''
各タスク用のtransformsたち
'''
import torchvision.transforms as transforms
from PIL import Image

# def get_transform(opt, grayscale=False, convert=True, method=transforms.InterpolationMode.BICUBIC):
def get_transform(opt, grayscale=False, convert=True, method=Image.BICUBIC): # for ist cluster
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.cop_size, method)))


    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    
    if 'flip' in opt.preprocess:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if convert:
        transform_list.append(transforms.ToTensor())
        if grayscale:
            transform_list.append(transforms.Normalize((0.5,),(0.5,)))
        else:
            transform_list.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
            
    return transforms.Compose(transform_list)

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