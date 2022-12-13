'''
torch.utils.data.Datasetを継承したDatasetたち
transform等が異なる
'''

from PIL import Image
from torch.utils.data import Dataset
import torch
import os.path as osp

def get_filepath_list(dataset_path:str, phase:str):
    template_path = osp.join('datasets', dataset_path, 'images' , '%s.jpg') # jpg以外も考えられる
    txtfile_path = osp.join('datasets', dataset_path, phase + '.txt')

    if not osp.exists(txtfile_path):
        raise NotImplementedError('dataset error')
    
    filepath_list = []
    for line in open(txtfile_path):
        file_id = line.strip()
        img_path = template_path % file_id
        filepath_list.append(img_path)
    
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