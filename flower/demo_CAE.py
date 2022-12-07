import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from model import CAE
from utils import make_filepath_list
from preprocessing import FlowerTransform, FlowerDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

autoEncoder = CAE().to(device)
autoEncoder.load_state_dict(torch.load('./weight/CAE_final.th'))
autoEncoder.eval()

batch_size = 8
_, _, test_dst_filepath_list, test_src_filepath_list = make_filepath_list()
transform = FlowerTransform(mean=0.5, std=0.5)
test_dataset = FlowerDataset(test_dst_filepath_list, test_src_filepath_list, transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

batch_iterator = iter(test_dataloader)
data = next(batch_iterator)
src, dst = data
src.to(device)
regen = autoEncoder(src)

fig = plt.figure(figsize=(12, 9))
for i in range(0, 5):

    plt.subplot(3, 5, i+1)
    plt.imshow(src[i][0].cpu().detach().numpy().transpose(1,2,0))

    plt.subplot(3, 5, 5+i+1)
    plt.imshow(dst[i][0].cpu().detach().numpy().transpose(1,2,0))

    plt.subplot(3, 5, 10+i+1)
    plt.imshow(regen[i][0].cpu().detach().numpy().trannspose(1,2,0))