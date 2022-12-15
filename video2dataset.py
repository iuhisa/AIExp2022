'''
convert 1 mp4 file to 1 dataset

'''

import os.path as osp
import cv2
from package.util import check_dir
import numpy as np
import argparse
from tqdm import tqdm

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create dataset based on 1 video file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('-w', '--width', type=int, default=-1, help='Cropping width. No cropping will be applied if its negative value.')
    parser.add_argument('--height', type=int, default=-1, help='Cropping heihgt. No cropping will be applied if its negative value.')
    parser.add_argument('-s', '--scale_width', type=int, default=-1, help='Scaling to fit specified width while keeping aspect ratio. No scaling is applied if its negative value. If both [-w/-h] and [--scale] are specified, cropping first, then scaling.')
    parser.add_argument('-f', '--fps', type=int, default=10, help='Frame rate of output images.')
    args = parser.parse_args()

    video_filename = args.input
    # target_size =  512 # 128, 256, 512

    dataset_name = args.dataset
    dataset_path = osp.join('datasets', dataset_name)

    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        exit(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out_frame_rate = int(args.fps)
    img_names = []

    if args.width > 0: # cropping
        crop_width = min(args.width, frame_width)
    else:
        crop_width = frame_width
    if args.height > 0:
        crop_height = min(args.height, frame_height)
    else:
        crop_height = frame_height

    if args.scale_width > 0: # scaling
        scale_width = args.scale_width
    else:
        scale_width = crop_width
    scale_height = int(crop_height/crop_width*scale_width)


    check_dir(osp.join(dataset_path, 'images'))
    for i in tqdm(range(frame_count)):
        if (i*out_frame_rate) % frame_rate != 0:continue
        ret, frame = cap.read()
        # if not ret: break
        img_name = 'img_{}'.format(str(i).zfill(3))

        cropped_frame = frame[frame_height//2 - crop_height//2:frame_height//2 + crop_height//2, frame_width//2 - crop_width//2:frame_width//2 + crop_width//2, :]
        resized_frame = cv2.resize(cropped_frame, (scale_width, scale_height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(dataset_path, 'images', img_name + '.jpg'), resized_frame)
        img_names.append(img_name)
        i += 1
    cap.release()

    name_list = '\n'.join(img_names)
    with open(osp.join(dataset_path, 'test.txt'), 'w') as f:
        f.write(name_list)