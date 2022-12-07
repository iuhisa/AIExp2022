import glob
import os.path

def make_filepath_list(root = './dataset'):
    train_dirpath = os.path.join(root, 'train')
    test_dirpath = os.path.join(root, 'test')

    train_dst_filepath_list = glob.glob(os.path.join(train_dirpath, 'dst', '*.jpg'))
    train_src_filepath_list = glob.glob(os.path.join(train_dirpath, 'src', '*.jpg'))
    test_dst_filepath_list = glob.glob(os.path.join(test_dirpath, 'dst', '*.jpg'))
    test_src_filepath_list = glob.glob(os.path.join(test_dirpath, 'src', '*.jpg'))
    
    return train_dst_filepath_list, train_src_filepath_list, test_dst_filepath_list, test_src_filepath_list

