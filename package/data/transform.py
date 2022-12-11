'''
各タスク用のtransformsたち
'''
import torchvision.transforms as transforms

def get_transform(opt, grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize))
    
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

