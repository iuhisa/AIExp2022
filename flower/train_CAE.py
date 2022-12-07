import time
from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt

from model import CAE
from utils import make_filepath_list
from preprocessing import FlowerTransform, FlowerDataset

# flower 画像のファイルパスリストを取得
train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list = make_filepath_list()

# test
print(train_dst_filepath_list)

# Datasetにする
transform = FlowerTransform(mean=0.5, std=0.5)
train_dataset = FlowerDataset(train_dst_filepath_list, train_src_filepath_list, transform)
test_dataset = FlowerDataset(test_dst_filepath_list, test_src_filepath_list, transform)

# Dataloaderにする
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelを作る
autoEncoder = CAE()

# 学習
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using: ' + device)

lr = 0.001
e_optim = torch.optim.Adam(autoEncoder.encoder.parameters(), lr)
d_optim = torch.optim.Adam(autoEncoder.decoder.parameters(), lr)

loss_fn = torch.nn.MSELoss(reduction='mean')

torch.backends.cudnn.benchmark = True
autoEncoder.to(device)

num_epochs = 1000
save_interval = 10 #10epochごとに評価、保存。
train_losses = []
val_losses = []
logs = []

for epoch in range(num_epochs):
    t_epoch_start = time.time()
    train_epoch_loss = 0
    val_epoch_loss = 0
    iteration = 0

    autoEncoder.train()
    print('-------------')
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-------------')
    print('（train）')

    for data in tqdm(train_dataloader):
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

        train_epoch_loss += loss.item()*batch_len
        iteration += 1
    
    t_epoch_finish = time.time()
    print('-------------')
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch, train_epoch_loss/batch_size))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
    
    autoEncoder.eval()
    print('-------------')
    print('（eval）')
    with torch.no_grad():
        for data in test_dataloader:
            dst, src = data
            dst.to(device)
            src.to(device)

            batch_len = len(dst)

            result = autoEncoder(dst)
            loss = loss_fn(src, result)
            val_epoch_loss += loss.item()*batch_len

    print('Loss:{:.4f}'.format(epoch, val_epoch_loss/batch_size))
    
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    log_epoch = {'epoch': epoch+1, 'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv("./result/log.csv")

    if (epoch + 1) % save_interval == 0:
        torch.save(autoEncoder.state_dict(), './weight/CAE_' + str(epoch + 1) + '.th')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = [a+1 for a in range(epoch)]
        ax.plot(x, train_losses, label='train loss')
        ax.plot(x, val_losses, label='val loss')
        plt.xlabel('epoch')
        plt.legend()
        fig.savefig('./result/loss_plot.pdf')

torch.save(autoEncoder.state_dict(), './weight/CAE_final.th')