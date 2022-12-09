'''
torch.utils.data.Datasetを継承したDatasetたち
transform等が異なる
'''

from PIL import Image
from torch.utils.data import Dataset

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