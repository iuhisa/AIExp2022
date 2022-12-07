import time
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from model import CAE
from utils import make_filepath_list, check_dir
from demo_CAE import demo
from preprocessing import FlowerTransform, FlowerDataset

# flower 画像のファイルパスリストを取得
train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list = make_filepath_list()

# Datasetにする
transform = FlowerTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
train_dataset = FlowerDataset(train_dst_filepath_list, train_src_filepath_list, transform)
test_dataset = FlowerDataset(test_dst_filepath_list, test_src_filepath_list, transform)

# Dataloaderにする
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelを作る
autoEncoder = CAE()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('ConvTranspose2d') != -1 or classname.find('Conv2d') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

autoEncoder.apply(weights_init)

# 固有ディレクトリの生成とチェック
IDENTITY = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
check_dir(osp.join('result', IDENTITY))
check_dir(osp.join('weight', IDENTITY))

# 学習
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using: ' + str(device))

lr = 0.001
e_optim = torch.optim.Adam(list(autoEncoder.encoder.parameters()), lr)
d_optim = torch.optim.Adam(list(autoEncoder.decoder.parameters()), lr)

loss_fn = torch.nn.MSELoss(reduction='mean')

torch.backends.cudnn.benchmark = True
autoEncoder.to(device)

num_epochs = 1000
save_interval = 10 #10epochごとに保存。
train_losses = []
val_losses = []
logs = []

for epoch in range(num_epochs):
    if epoch % save_interval == 0: # 10epochに一回はモデルを保存(epoch == 0のときも保存)
        torch.save(autoEncoder.state_dict(), osp.join('weight', IDENTITY, f'CAE_{epoch}.th'))
    t_epoch_start = time.time()
    train_epoch_loss = 0
    val_epoch_loss = 0
    iteration = 0

    ###
    ### train
    ###
    print('-------------')
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-------------')
    print('（train）')
    autoEncoder.train()
    for data in tqdm(train_dataloader):
        dst, src = data
        dst = dst.to(device)
        src = src.to(device)

        batch_len = len(dst)

        result = autoEncoder(src)
        loss = loss_fn(dst, result)

        e_optim.zero_grad()
        d_optim.zero_grad()
        loss.backward()
        e_optim.step()
        d_optim.step()

        train_epoch_loss += loss.item()*batch_len # 誤差は全部足す
        iteration += 1
    
    t_epoch_finish = time.time()
    train_epoch_loss /= len(train_dataset) # 画像1枚あたりの誤差に変換
    print('-------------')
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch+1, train_epoch_loss))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

    ###
    ### eval
    ###
    print('-------------')
    print('（eval）')
    autoEncoder.eval()
    with torch.no_grad():
        for data in test_dataloader:
            dst, src = data
            dst = dst.to(device)
            src = src.to(device)

            batch_len = len(dst)

            result = autoEncoder(src)
            loss = loss_fn(dst, result)
            val_epoch_loss += loss.item()*batch_len

    val_epoch_loss /= len(test_dataset)
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch+1, val_epoch_loss))
    
    # lossをlogに保存
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    log_epoch = {'epoch': epoch+1, 'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv(osp.join('result', IDENTITY, 'log.csv'))

    # lossをプロットして保存
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [a+1 for a in range(epoch+1)]
    ax.plot(x, train_losses, label='train loss')
    ax.plot(x, val_losses, label='val loss')
    plt.xlabel('epoch')
    plt.legend()
    fig.savefig(osp.join('result', IDENTITY, 'loss_plot.pdf'))

    # 変換した画像も保存
    demo(autoEncoder=autoEncoder, device=device, out_path=osp.join('result', IDENTITY, f'demo_{epoch+1}.png'))

torch.save(autoEncoder.state_dict(), osp.join('weight', IDENTITY, 'CAE_final.th'))