'''
convert 1 mp4 file to 1 dataset

'''

import os.path as osp
import cv2
from package.util import check_dir
import numpy as np

video_filename = 'mountain_video.mp4'

dataset_name = video_filename.split('.')[0]
dataset_path = osp.join('datasets', dataset_name)
check_dir(osp.join(dataset_path, 'images'))

cap = cv2.VideoCapture(video_filename)
if not cap.isOpened():
    exit(0)

i = 0
img_names = []

while True:
    ret, frame = cap.read()
    if not ret: break
    img_name = 'img_{}.jpg'.format(str(i).zfill(3))
    cv2.imwrite(osp.join(dataset_path, 'images', img_name), frame)
    img_names.append(img_name)
    i += 1
cap.release()

name_list = '\n'.join(img_names)
with open(osp.join(dataset_path, 'test.txt'), 'w') as f:
    f.write(name_list)