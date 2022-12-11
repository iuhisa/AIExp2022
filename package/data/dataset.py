'''
torch.utils.data.Datasetを継承したDatasetたち
transform等が異なる
'''

from PIL import Image
from torch.utils.data import Dataset
import os.path as osp

def get_filepath_list(dataset_path:str, phase:str):
    template_path = osp.join(dataset_path, 'images' , '%s.jpg') # jpg以外も考えられる
    txtfile_path = osp.join(dataset_path, phase + '.txt')

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
        super(self, SingleDataset).__init__()
        self.paths = dataset_path_list
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

'''
class FlowerDataset(Dataset):
    def __init__(self, dst_path_list, src_path_list, transform):
        self.dst_path_list = dst_path_list
        self.src_path_list = src_path_list
        self.transform = transform

    def __len__(self):
        return len(self.dst_path_list)

    def __getitem__(self, index):
        dst_image_path = self.dst_path_list[index]
        dst_image = Image.open(dst_image_path)
        dst_image_transformed = self.transform(dst_image)
        src_image_path = self.src_path_list[index]
        src_image = Image.open(src_image_path)
        src_image_transformed = self.transform(src_image)
        return dst_image_transformed, src_image_transformed
'''