'''
指定されたdataset, phase['train', 'val', ...]からtorch.utils.data.DataLoaderを作成する。
transformerのパラメータなどを指定できるようにしないといけない
タスクによって生成するdataloaderが異なるかも。
'''

from . import dataset
from . import transform
# get_train_dataloader() -> dataloader
# get_test_dataloader() -> dataloader

# 複数ドメインのdataloaderを用意？
def get_dataloader(dataset_name:str, batch_size:int, phase:str):
    '''
    parameters
    ----------
        dataset_name : name of dataset; as directory name which is child-dir of ./datasets
        batch_size : size of batch
        phase : 'train', 'val' or 'test'
    '''
    pass
