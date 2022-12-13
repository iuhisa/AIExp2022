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

# 複数ドメインのdataloaderを用意
def get_unpair_dataloader(opt):
    '''
    parameters
    ----------
        dataset_name : name of dataset; as directory name which is child-dir of ./datasets
        batch_size : size of batch
        phase : 'train', 'val' or 'test'
    '''
    if opt.max_dataset_size == float('inf'):
        A_paths = dataset.get_filepath_list(opt.A_dataroot, opt.phase)
        B_paths = dataset.get_filepath_list(opt.B_dataroot, opt.phase)
    else:
        A_paths = dataset.get_filepath_list(opt.A_dataroot, opt.phase)[:opt.max_dataset_size]
        B_paths = dataset.get_filepath_list(opt.B_dataroot, opt.phase)[:opt.max_dataset_size]

    trans = transform.get_transform(opt)

    if opt.A_datatype == 'isolated':
        A_dataset = dataset.SingleDataset(A_paths, trans)
    elif opt.A_datatype == 'sequential':
        A_dataset = dataset.SequentialDataset(A_paths, trans, n=opt.sequential_len)

    if opt.B_datatype == 'isolated':
        B_dataset = dataset.SingleDataset(B_paths, trans)
    elif opt.B_datatype == 'sequential':
        B_dataset = dataset.SequentialDataset(B_paths, trans, n=opt.sequential_len)

    print('A datset num: {}, B dataset num: {}'.format(len(A_dataset), len(B_dataset)))

    A_dataloader = DataLoader(A_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    B_dataloader = DataLoader(B_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    return A_dataloader, B_dataloader

def get_dataloader(opt, domain='A'):
    if domain == 'A':
        dataroot = opt.A_dataroot
        datatype = opt.A_datatype
    elif domain == 'B':
        dataroot = opt.B_dataroot
        datatype = opt.B_datatype

    if opt.max_dataset_size == float('inf'):
        paths = dataset.get_filepath_list(dataroot, opt.phase)
    else:
        paths = dataset.get_filepath_list(dataroot, opt.phase)[:opt.max_dataset_size]
    
    trans = transform.get_transform(opt)

    if datatype == 'isolated':
        _dataset = dataset.SingleDataset(paths, trans)
    elif datatype == 'sequential':
        _dataset = dataset.SequentialDataset(paths, trans, n=opt.sequential_len)

    dataloader = DataLoader(_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    return dataloader