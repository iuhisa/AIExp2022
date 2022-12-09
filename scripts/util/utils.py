import glob
import os.path
import os
import torch.cuda as cuda

def make_filepath_list(root = 'dataset'):
    train_dirpath = os.path.join(root, 'train')
    test_dirpath = os.path.join(root, 'test')

    train_dst_filepath_list = glob.glob(os.path.join(train_dirpath, 'dst', '*.jpg'))
    train_src_filepath_list = glob.glob(os.path.join(train_dirpath, 'src', '*.jpg'))
    test_dst_filepath_list = glob.glob(os.path.join(test_dirpath, 'dst', '*.jpg'))
    test_src_filepath_list = glob.glob(os.path.join(test_dirpath, 'src', '*.jpg'))
    
    return train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list

# ディレクトリがなければ作る。再帰的に作れる。
def check_dir(path):
    os.makedirs(path, exist_ok=True)

def get_gpu_list():
    if cuda.is_available():
        ret = list(range(cuda.current_device(), cuda.current_device() + cuda.device_count()))
    else:
        ret = []
    return ret