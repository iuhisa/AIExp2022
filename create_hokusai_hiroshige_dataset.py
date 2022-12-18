import os.path as osp
import os
import cv2
from package.util import check_dir, clip_image
from tqdm import tqdm
import numpy as np

NUM_FILTER = 100

def main():
    
    if not osp.exists(osp.join('datasets', 'ukiyoe_high_resolution')):
        print('ukiyoe_high_resolutionデータセットがありません。\n先にhttps://www.kaggle.com/datasets/kengoichiki/the-metropolitan-museum-of-art-ukiyoe-dataset?resource=downloadよりデータをダウンロードして解凍して下さい。\nukiyoe_high_resolution/archiveとなる様にして下さい。')
        return
    dir_path_ukiyoe_high_resolution = osp.join('datasets', 'ukiyoe_high_resolution', 'archive', 'images')
    print('katsushika_hokusaiデータセットを作成します')
    dir_path_hokusai = osp.join('datasets', 'katsushika_hokusai','images')
    check_dir(dir_path_hokusai)
    for i, file_path in enumerate(tqdm(os.listdir(osp.join(dir_path_ukiyoe_high_resolution, 'Katsushika_Hokusai')))):
        img = cv2.imread(osp.join(dir_path_ukiyoe_high_resolution, 'Katsushika_Hokusai', file_path))
        h, w = img.shape[:2]
        img = clip_image(img, min(h*0.8, w*0.8), min(h*0.8, w*0.8))
        if type(img)==np.ndarray:
            img = cv2.resize(img, (1000, 1000))
            cv2.imwrite(osp.join(dir_path_hokusai, f'{i}.jpg'), img)
    
    print('utagawa_hiroshigeデータセットを作成します')
    dir_path_hiroshige = osp.join('datasets', 'utagawa_hiroshige','images')
    check_dir(dir_path_hiroshige)
    for i, file_path in enumerate(tqdm(os.listdir(osp.join(dir_path_ukiyoe_high_resolution, 'Utagawa_Hiroshige')))):
        img = cv2.imread(osp.join(dir_path_ukiyoe_high_resolution, 'Utagawa_Hiroshige', file_path))
        h, w = img.shape[:2]
        img = clip_image(img, min(h*0.8, w*0.8), min(h*0.8, w*0.8))
        if type(img)==np.ndarray:
            img = cv2.resize(img, (1000, 1000))
            cv2.imwrite(osp.join(dir_path_hiroshige, f'{i}.jpg'), img)
    
    for j, file_path in enumerate(tqdm(os.listdir(osp.join(dir_path_ukiyoe_high_resolution, 'Utagawa_Hiroshige_II')))):
        img = cv2.imread(osp.join(dir_path_ukiyoe_high_resolution, 'Utagawa_Hiroshige_II', file_path))
        h, w = img.shape[:2]
        img = clip_image(img, min(h*0.8, w*0.8), min(h*0.8, w*0.8))
        if type(img)==np.ndarray:
            img = cv2.resize(img, (1000, 1000))
            cv2.imwrite(osp.join(dir_path_hiroshige, f'{i}.jpg'), img)

if __name__ == "__main__":
    main()