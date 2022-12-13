import cv2
from glob import glob
import os
import os.path as osp
from tqdm import tqdm

src_dir = 'datasets/nature_video'
dst_dir = 'datasets/nature_stylization'

img_paths = glob(osp.join(src_dir, 'images', '*.jpg'))
dst_path = osp.join(dst_dir, 'images')

for path in tqdm(img_paths):
    src = cv2.imread(path)
    name = os.path.splitext(os.path.basename(path))[0].split('_')[0]
    number = os.path.splitext(os.path.basename(path))[0].split('_')[1]
    dst = cv2.stylization(src, sigma_s=60, sigma_r=0.6)
    cv2.imwrite(osp.join(dst_path, name)+'-stylization_'+number+'.jpg', dst)