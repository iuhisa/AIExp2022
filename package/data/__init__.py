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

    A_dataloader = get_dataloader(opt, 'A')
    B_dataloader = get_dataloader(opt, 'B')

    return A_dataloader, B_dataloader

def get_paired_dataloader(opt):
    dataloader = get_dataloader(opt)
    return dataloader

def get_dataloader(opt, domain='A'):
    if domain == 'A':
        dataroot = opt.A_dataroot
        datatype = opt.A_datatype
    elif domain == 'B':
        dataroot = opt.B_dataroot
        datatype = opt.B_datatype

    paths = dataset.get_filepath_list(dataroot, opt.phase, list_len_max=opt.max_dataset_size)
    trans = transform.get_transform(opt, domain=domain)

    if datatype == 'isolated':
        _dataset = dataset.SingleDataset(paths, trans)
    elif datatype == 'sequential':
        _dataset = dataset.SequentialDataset(paths, trans, n=opt.sequential_len)
    elif datatype == 'aligned':
        _dataset = dataset.AlignedDataset(opt)


    print('Domain: {}, Dataset num: {}'.format(domain, len(_dataset)))
    dataloader = DataLoader(_dataset, opt.batch_size, shuffle=opt.isTrain, num_workers=opt.num_threads)
    return dataloader