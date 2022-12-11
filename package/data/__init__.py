'''
指定されたdataset, phase['train', 'val', ...]からtorch.utils.data.DataLoaderを作成する。
transformerのパラメータなどを指定できるようにしないといけない
タスクによって生成するdataloaderが異なるかも。
'''
from torch.utils.data import DataLoader

from . import dataset
from . import transform
import os.path as osp
# get_train_dataloader() -> dataloader
# get_test_dataloader() -> dataloader

# 複数ドメインのdataloaderを用意？
def get_unpair_dataloader(opt):
    '''
    parameters
    ----------
        dataset_name : name of dataset; as directory name which is child-dir of ./datasets
        batch_size : size of batch
        phase : 'train', 'val' or 'test'
    '''
    A_paths = dataset.get_filepath_list(opt.A_dataroot, opt.phase)[:opt.max_dataset_size]
    B_paths = dataset.get_filepath_list(opt.B_dataroot, opt.phase)[:opt.max_dataset_size]
    trans = transform.get_transform(opt)
    A_dataset = dataset.SingleDataset(A_paths, trans)
    B_dataset = dataset.SingleDataset(B_paths, trans)
    A_dataloader = DataLoader(A_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    B_dataloader = DataLoader(B_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    return A_dataloader, B_dataloader