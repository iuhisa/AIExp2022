import os
import os.path as osp
from glob import glob

def make_img_list(dataset_root_dir):
    '''
    dataset_root_dir e.g. 'datasets/ukiyoe'
    '''
    train, val, test = 8, 1, 1
    total = train + val + test
    # dataset_root_dirs = [f for f in glob(osp.join('datasets','*')) if osp.isdir(f)]
    # print(dataset_root_dirs)
    root_dir = dataset_root_dir
    img_paths = glob(osp.join(root_dir, 'images', '*.jpg'))
    img_num = len(img_paths)
    i1 = int(img_num*train/total)
    i2 = int(img_num*(train+val)/total)

    train_paths = [a.split(os.sep)[-1].split('.')[0] for a in img_paths[:i1]]
    val_paths = [a.split(os.sep)[-1].split('.')[0] for a in img_paths[i1: i2]]
    test_paths = [a.split(os.sep)[-1].split('.')[0] for a in img_paths[i2:]]

    train_list = '\n'.join(train_paths)
    val_list = '\n'.join(val_paths)
    test_list = '\n'.join(test_paths)
    with open(osp.join(root_dir, 'train.txt'), 'w') as f:
        f.write(train_list)
    with open(osp.join(root_dir, 'test.txt'), 'w') as f:
        f.write(test_list)
    with open(osp.join(root_dir, 'val.txt'), 'w') as f:
        f.write(val_list)

if __name__=='__main__':
    make_img_list()