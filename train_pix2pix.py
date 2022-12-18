import time
import torch

from package.model import get_model
from package.util import visualize
from package.data import get_paired_dataloader
from package.options.train_options import TrainOptions

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    dataloader = get_paired_dataloader(opt)
    # print('The number of training images = %d, %d' % (len(A_dataloader)*opt.batch_size, len(B_dataloader)*opt.batch_size))
    batch_multiplier = opt.batch_multiplier
    print(f'Virtual Batchsize: {batch_multiplier * opt.batch_size}')

    model = get_model(opt)
    model.setup(opt)
    visualizer = visualize.Visualizer(opt, model.loss_names)

    total_iters = 0
    batch_count = batch_multiplier

    for epoch in range(opt.epoch_count, opt.n_epochs + 1):
        epoch_start_time = time.time()

        for data in dataloader:
            total_iters += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # model.forward()
            # model.backward()
            losses = model.get_current_losses()
            visualizer.store_loss(losses)
            batch_count -= 1

        if epoch % opt.save_epoch_interval == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch) #ファイルが大きすぎ
            visualizer.plot_loss()

            visualizer.save_images(model, opt=opt, epoch=epoch)
            # visualizer.save_imgs(model.real_A[:save_n], model.fake_B[:save_n], model.rec_A[:save_n], epoch=epoch, id='AtoB')
            # visualizer.save_imgs(model.real_B[:save_n], model.fake_A[:save_n], model.rec_B[:save_n], epoch=epoch, id='BtoA')

        print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.n_epochs, time.time() - epoch_start_time))
        visualizer.save_loss(epoch)
        model.update_learning_rate()
