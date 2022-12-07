import time
import torch

from model import CAE
from utils import make_filepath_list
from preprocessing import FlowerTransform, FlowerDataset

# flower 画像のファイルパスリストを取得
train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list = make_filepath_list()

# Datasetにする
transform = FlowerTransform()
train_dataset = FlowerDataset(train_dst_filepath_list, train_src_filepath_list, transform)
test_dataset = FlowerDataset(test_dst_filepath_list, test_src_filepath_list, transform)

# Dataloaderにする
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Modelを作る
autoEncoder = CAE()

# 学習
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using: ' + device)

lr = 0.001
e_optim = torch.optim.Adam(autoEncoder.encoder.parameters(), lr)
d_optim = torch.optim.Adam(autoEncoder.decoder.parameters(), lr)

loss_fn = torch.nn.MSELoss(reduction='mean')

autoEncoder.to(device)
torch.backends.cudnn.benchmark = True
autoEncoder.train()

num_epochs = 100

for epoch in range(num_epochs):
    t_epoch_start = time.time()
    epoch_loss = 0
    iteration = 0
    print('-------------')
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-------------')
    print('（train）')

    for data in train_dataloader:
        dst, src = data
        dst.to(device)
        src.to(device)

        batch_len = len(dst)

        result = autoEncoder(dst)
        loss = loss_fn(src, result)

        e_optim.zero_grad()
        d_optim.zero_grad()
        loss.backward()
        e_optim.step()
        d_optim.step()

        epoch_loss += loss.detach().item()*batch_len
        iteration += 1
    
    t_epoch_finish = time.time()
    print('-------------')
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch, epoch_loss/batch_size))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
