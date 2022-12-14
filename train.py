import time
import torch

from package.model import get_model
from package.util import visualize
from package.data import get_unpair_dataloader
from package.options.train_options import TrainOptions

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    A_dataloader, B_dataloader = get_unpair_dataloader(opt)
    # print('The number of training images = %d, %d' % (len(A_dataloader)*opt.batch_size, len(B_dataloader)*opt.batch_size))

    model = get_model(opt)
    model.setup(opt)
    # print(model.loss_names)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()
        # losses = None
        # visualizer.reset()

        for A_data, B_data in zip(A_dataloader, B_dataloader):
            total_iters += opt.batch_size
            data = {'A':A_data, 'B':B_data}
            model.set_input(data)
            model.optimize()
            losses = model.get_current_losses()
            # lossesのkeyごとに足す。
            visualizer.store_loss(losses)

        if epoch % opt.save_epoch_interval == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            # model.save_networks(epoch) ファイルが大きすぎ
            visualizer.plot_loss()

            # modelにimageを保存させる機能を持たせるのが吉か。
            save_n = opt.save_image_num
            visualizer.save_images(model, epoch=epoch)
            # visualizer.save_imgs(model.real_A[:save_n], model.fake_B[:save_n], model.rec_A[:save_n], epoch=epoch, id='AtoB')
            # visualizer.save_imgs(model.real_B[:save_n], model.fake_A[:save_n], model.rec_B[:save_n], epoch=epoch, id='BtoA')

        print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.n_epochs, time.time() - epoch_start_time))
        visualizer.save_loss(epoch)
        model.update_learning_rate()

'''legacy
# GPU or CPU
gpu_ids = get_gpu_list()
device = torch.device('cuda' if len(gpu_ids) > 0 else 'cpu')
print('using: ' + str(device) + ', ' + str(gpu_ids))

# flower 画像のファイルパスリストを取得
train_dst_filepath_list, train_src_filepath_list, val_dst_filepath_list, val_src_filepath_list = make_filepath_list()

# Datasetにする
# transform = FlowerTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# train_dataset = FlowerDataset(train_dst_filepath_list, train_src_filepath_list, transform)
# val_dataset = FlowerDataset(val_dst_filepath_list, val_src_filepath_list, transform)
# train_dataset_len = len(train_dataset)
# val_dataset_len = len(val_dataset)

# Dataloaderにする
batch_size = 64 # 1080Tiは64が限界
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = get_dataloader('flower', batch_size, 'train')
val_dataloader = get_dataloader('flower', batch_size, 'val')
train_dataset_len = len(train_dataloader)
val_dataset_len = len(val_dataloader)

# Modelを作る
model = get_model('CAE', None)

# 固有ディレクトリの生成とチェック
IDENTITY = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
check_dir(osp.join('result', IDENTITY))
check_dir(osp.join('weight', IDENTITY))

# 学習
# lr = 0.001
# e_optim = torch.optim.Adam(list(model.encoder.parameters()), lr)
# d_optim = torch.optim.Adam(list(model.decoder.parameters()), lr)
# loss_fn = torch.nn.MSELoss(reduction='mean')

# 要変更
# if len(gpu_ids) > 0: # GPUが使えるときは並列処理できるやつに変換
#     model = nn.DataParallel(model, device_ids=gpu_ids)
# model.to(device)
torch.backends.cudnn.benchmark = True

num_epochs = 100
save_interval = 10 # N epochごとに保存。
train_losses = []
val_losses = []
logs = []

for epoch in range(num_epochs):
    if epoch % save_interval == 0: # N epochに一回モデルを保存(epoch == 0のときも保存) & 結果をplot
        pass
        # model.save(osp.join('weight', IDENTITY, f'CAE_{epoch}.th'))
        # torch.save(model.state_dict(), osp.join('weight', IDENTITY, f'CAE_{epoch}.th'))
        # visualize.save(autoEncoder=model, device=device, out_path=osp.join('result', IDENTITY, f'demo_{epoch}.png'))
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
    model.train()
    for data in tqdm(train_dataloader):
        dst, src = data
        dst = dst.to(device)
        src = src.to(device)

        batch_len = len(dst)

        model.set_input(src, dst)
        model.forward()
        model.backward()
        model.optimize()
        # result = model(src)
        # loss = loss_fn(dst, result)

        # e_optim.zero_grad()
        # d_optim.zero_grad()
        # loss.backward()
        # e_optim.step()
        # d_optim.step()

        train_epoch_loss += loss.item()*batch_len # 誤差は全部足す
        iteration += 1
    
    t_epoch_finish = time.time()
    train_epoch_loss /= train_dataset_len # 画像1枚あたりの誤差に変換
    print('-------------')
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch+1, train_epoch_loss))
    print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

    ###
    ### eval
    ###
    print('-------------')
    print('（eval）')
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            dst, src = data
            dst = dst.to(device)
            src = src.to(device)

            batch_len = len(dst)

            model.set_input(src, dst)
            model.forward()
            loss = model.get_loss()
            # result = model(src)
            # loss = loss_fn(dst, result)
            val_epoch_loss += loss.item()*batch_len

    val_epoch_loss /= val_dataset_len
    print('epoch {} || Epoch_Loss:{:.4f}'.format(epoch+1, val_epoch_loss))
    
    # lossをlogに保存
    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    log_epoch = {'epoch': epoch+1, 'train_loss': train_epoch_loss, 'val_loss': val_epoch_loss}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv(osp.join('result', IDENTITY, 'log.csv'), index=False)

    # lossをプロットして保存
    fig = plt.figure(num=1, clear=True) # memory leak 対策
    ax = fig.subplots()
    x = [a+1 for a in range(epoch+1)]
    ax.plot(x, train_losses, label='train loss')
    ax.plot(x, val_losses, label='val loss')
    ax.set_xlabel('epoch')
    ax.legend()
    fig.savefig(osp.join('result', IDENTITY, 'loss_plot.pdf'))

# torch.save(model.state_dict(), osp.join('weight', IDENTITY, 'CAE_final.th'))
'''