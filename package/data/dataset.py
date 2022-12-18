'''
torch.utils.data.Datasetを継承したDatasetたち
transform等が異なる
'''

from PIL import Image
from torch.utils.data import Dataset
import torch
import os.path as osp
from . import transform

def get_filepath_list(dataset_path:str, phase:str, list_len_max:int):
    template_path = osp.join('datasets', dataset_path, 'images' , '%s.jpg') # jpg以外も考えられる
    txtfile_path = osp.join('datasets', dataset_path, phase + '.txt')

    if not osp.exists(txtfile_path):
        raise NotImplementedError('dataset error')
    
    i = 1
    filepath_list = []
    for line in open(txtfile_path):
        if i > list_len_max: break
        file_id = line.strip()
        img_path = template_path % file_id
        filepath_list.append(img_path)
        i += 1
    
    return filepath_list

class SingleDataset(Dataset):
    def __init__(self, dataset_path_list, transform):
        super(SingleDataset, self).__init__()
        self.paths = dataset_path_list
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

class SequentialDataset(Dataset):
    def __init__(self, dataset_path_list, transform, n=3):
        super(SequentialDataset, self).__init__()
        self.paths = dataset_path_list
        self.transform = transform
        self.n = n

    def __len__(self):
        return len(self.paths) - self.n + 1

    def __getitem__(self, index):
        paths = self.paths[index: index+self.n]
        imgs = [Image.open(path).convert('RGB') for path in paths]
        imgs = [self.transform(img) for img in imgs]
        return torch.stack(imgs)

# ちょい特殊
class AlignedDataset(Dataset):
    def __init__(self, opt):
        super(AlignedDataset, self).__init__()
        
        A_path_list = get_filepath_list(opt.A_dataroot, opt.phase, list_len_max=opt.max_dataset_size)
        self.A_paths = A_path_list
        B_path_list = get_filepath_list(opt.B_dataroot, opt.phase, list_len_max=opt.max_dataset_size)
        self.B_paths = B_path_list
        self.A_transform = transform.get_transform(opt, domain='A', grayscale=(opt.input_nc == 1))
        self.B_transform = transform.get_transform(opt, domain='B', grayscale=(opt.output_nc == 1))
        assert(len(self.A_paths) == len(self.B_paths))

    def __len__(self):
        return len(self.A_paths)
    
    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_img = self.A_transform(A_img)
        B_img = self.B_transform(B_img)
        return {'A': A_img, 'B': B_img}