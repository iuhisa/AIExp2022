from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class FlowerTransform():
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.data_transform(img)

class FlowerDataset(data.Dataset):
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