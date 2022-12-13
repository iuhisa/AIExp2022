'''
1.datasetをダウンロードする
2.datasetに含まれる画像のPathを列挙する(Pathリスト)
3.Pathリストを train, val, test に分けて、train.txt, val.txt, test.txt などに保存する(割合は指定する)
(4. datasetをロードするときはこのPathリストを見て、ロードする。)

以下は、ドメイン分けされたunpairなデータセットの例
pairのあるデータセットだったり、タスクによって構成が変わり得る

./ ┬ datasets ┬ flower_pansy ┬ images ──┬ 001.jpg
               │              │            ├ 002.jpg
               │              ├ train.txt
               │              ├ val.txt 要る???
               │              └ test.txt
               ├ flower_dandelion ┬ ...

'''

import argparse
from glob import glob
import os
import os.path as osp
from package.util import check_dir
import requests
import shutil
from tqdm import tqdm
import zipfile
import cv2
import random
from transform_image import transform_image

DATASETS = {
    'ukiyoe': {
        'url': 'http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/ukiyoe2photo.zip',
        'desc': 'これをダウンロードすることでphoto_ukiyoeデータセットも作成されます'} ,
    'photo_ukiyoe': {'desc': 'ukiyoeデータセットの作成に伴って作成されます'},
    'vangogh': {
        'url': 'http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip',
        'desc': 'これをダウンロードすることでphoto_vangoghデータセットも作成されます'} ,
    'photo_vangogh': {'desc': 'vangoghデータセットの作成に伴って作成されます'},
    'nature_video': {'desc': '素材サイトのPexelsから、natureに該当する動画を連番画像で出力します'},
    'ukiyoe_video': {'desc': 'ukiyoeデータセットが必要です。インターネット接続は不要です'}
}

PEXELS_API = '563492ad6f91700001000001c85472c49d3a4c189a1ed7baa64e4ae5'
NUM_FILTER = 10 #ukiyoe_video作成時に何回フィルタをかけるか

random.seed(5555)

def make_img_list(dataset_root_dir):
    '''
    Make three text files 'train.txt', 'val.txt' and 'test.txt'.

    Used for dataset of isolated images.

    Each files contain names of image files to use in train, validation and test mode.
    
    Parameters
    ---------------------------------------
        dataset_root_dir: path to dataset root. e.g. 'datasets/ukiyoe'
    '''
    train, val, test = 8, 1, 1
    total = train + val + test
    # dataset_root_dirs = [f for f in glob(osp.join('datasets','*')) if osp.isdir(f)]
    # print(dataset_root_dirs)
    root_dir = dataset_root_dir
    img_paths = glob(osp.join(root_dir, 'images', '*.jpg'))

    # ファイルの名前だけ取り出し -> sort -> randomに並び替え
    img_paths = [a.split(os.sep)[-1].split('.')[0] for a in img_paths]
    img_paths.sort()
    random.shuffle(img_paths)
    #

    img_num = len(img_paths)
    i1 = int(img_num*train/total)
    i2 = int(img_num*(train+val)/total)

    train_paths = img_paths[:i1]
    val_paths = img_paths[i1: i2]
    test_paths = img_paths[i2:]

    train_list = '\n'.join(train_paths)
    val_list = '\n'.join(val_paths)
    test_list = '\n'.join(test_paths)
    
    with open(osp.join(root_dir, 'train.txt'), 'w') as f:
        f.write(train_list)
    with open(osp.join(root_dir, 'test.txt'), 'w') as f:
        f.write(test_list)
    with open(osp.join(root_dir, 'val.txt'), 'w') as f:
        f.write(val_list)

def remove_underscore_and_numbers(filename):
    return os.path.basename(filename).split('_')[0]

def make_vdo_list(dataset_root_dir):
    '''
    Make three text files named 'train.txt', 'val.txt' and 'test.txt'.

    Used for video-driven dataset.

    Each files contain names of image files to use in train, validation and test mode.
    
    Parameters
    ---------------------------------------
        dataset_root_dir: path to dataset root. e.g. 'datasets/nature_video'
    '''
    train, val, test = 8, 1, 1
    total = train + val + test

    root_dir = dataset_root_dir
    img_paths = glob(osp.join(root_dir, 'images', '*.jpg'))
    img_paths.sort()

    ####################################################################
    ##
    ## txtファイルに識別子のみを記載する場合
    ##
    # random.shuffle(img_paths)

    # img_paths_remove_underscores = list(map(remove_underscore_and_numbers, img_paths))
    # img_paths_remove_duplicates = list(dict.fromkeys(img_paths_remove_underscores))

    # img_num = len(img_paths_remove_duplicates)
    # i1 = int(img_num*train/total)
    # i2 = int(img_num*(train+val)/total)

    # train_paths = img_paths_remove_duplicates[:i1]
    # val_paths = img_paths_remove_duplicates[i1:i2]
    # test_paths = img_paths_remove_duplicates[i2:]

    ######################################################################
    ##
    ## txtファイルに連番も記載する場合
    ##
    id2names = dict() # 'hoge/fuga/{id}_{number}.jpg'に対して、 同じidのファイル名を1つの配列に格納。

    
    def get_filename(path): # pathから拡張子を抜いてファイル名だけ取り出す
        return osp.splitext(osp.basename(path))[0]

    for img_path in img_paths:
        img_id = remove_underscore_and_numbers(img_path)
        if not img_id in id2names.keys(): # 初めて見たidなら、配列を定義
            id2names[img_id] = []
        
        id2names[img_id].append(get_filename(img_path))

    ids = list(id2names.keys())
    ids.sort()
    for key in id2names.keys():
        id2names[key].sort()

    img_id_num = len(ids)
    i1 = int(img_id_num*train/total)
    i2 = int(img_id_num*(train+val)/total)

    train_paths = []
    val_paths = []
    test_paths = []
    for key_train in ids[:i1]:
        names = id2names[key_train]
        train_paths += names
    for key_val in ids[i1:i2]:
        names = id2names[key_val]
        val_paths += names
    for key_test in ids[i2:]:
        names = id2names[key_test]
        test_paths += names
    ##
    ##
    ##
    #################################################################

    train_list = '\n'.join(train_paths)
    val_list = '\n'.join(val_paths)
    test_list = '\n'.join(test_paths)
    with open(osp.join(root_dir, 'train.txt'), 'w') as f:
        f.write(train_list)
    with open(osp.join(root_dir, 'test.txt'), 'w') as f:
        f.write(test_list)
    with open(osp.join(root_dir, 'val.txt'), 'w') as f:
        f.write(val_list)

def download_berkeley_A2B(A: str, B: str):
    print(f'{A}ディレクトリ、{B}_{A}ディレクトリを作成します')
    if osp.exists(f'datasets/{A}') or osp.exists(f'datasets/{B}_{A}'):
        print(f'既にデータが存在しています。データを再ダウンロードしたい場合は{A}ディレクトリと{B}_{A}を削除してください')
        return
    dir_path_A = osp.join('datasets', f'{A}')
    dir_path_B = osp.join('datasets', f'{B}_{A}')
    download_file_path = osp.join('datasets', f'{A}', f'{A}.zip')
    check_dir(dir_path_A+'/images')
    check_dir(dir_path_B+'/images')
    
    print(f'{A}データセットをダウンロードします')
    file_url = DATASETS[f'{A}']['url']
    file_size = int(requests.head(DATASETS[f'{A}']['url']).headers["content-length"])
    res = requests.get(file_url, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)
    with open(download_file_path, 'wb') as file:
        for chunk in res.iter_content(chunk_size=8192):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    
    print('ディレクトリを再構築中です')
    zp = zipfile.ZipFile(download_file_path, 'r')
    zp.extractall(dir_path_A)
    zp.close()
    
    for file_path in os.listdir(dir_path_A+f'/{A}2{B}/testA'):
        shutil.move(dir_path_A+f'/{A}2{B}/testA/{file_path}', dir_path_A+f'/images/{file_path}')
    for file_path in os.listdir(dir_path_A+f'/{A}2{B}/trainA'):
        shutil.move(dir_path_A+f'/{A}2{B}/trainA/{file_path}', dir_path_A+f'/images/{file_path}')
    for file_path in os.listdir(dir_path_A+f'/{A}2{B}/testB'):
        shutil.move(dir_path_A+f'/{A}2{B}/testB/{file_path}', dir_path_B+f'/images/{file_path}')
    for file_path in os.listdir(dir_path_A+f'/{A}2{B}/trainB'):
        shutil.move(dir_path_A+f'/{A}2{B}/trainB/{file_path}', dir_path_B+f'/images/{file_path}')
    
    make_img_list(f'datasets/{A}')
    make_img_list(f'datasets/{B}_{A}')    

    print('不要なファイル・ディレクトリを削除します')
    shutil.rmtree(dir_path_A+f'/{A}2{B}')
    os.remove(download_file_path)

def download_nature_video():
    # 定数
    query = 'nature'
    results_per_page = 80
    page_numbers = 10 # 80*10=800個の動画をダウンロード
    download_size = 'small'
    max_frames = 200 # 連番画像の最大書き出し数（10FPS×200枚で最大20秒）
    target_fps = 10 # 10FPSに変換
    target_size = 256 # 256*256にリサイズ

    print('nature_videoディレクトリを作成します')
    if osp.exists('datasets/nature_video'):
        print('既にデータが存在しています。データを再ダウンロードしたい場合はnature_videoディレクトリを削除してください')
        return
    dir_path_nature_video = osp.join('datasets', 'nature_video')
    check_dir(dir_path_nature_video+'/images')

    print('Pexelsからnature_videoのダウンロードを開始します')
    PEXELS_AUTHORIZATION = {"Authorization": PEXELS_API}

    for i in range(1, page_numbers+1):
        url = "https://api.pexels.com/videos/search?query={}&size={}&per_page={}&page={}".format(query, download_size, results_per_page, i)
        request = requests.get(url, timeout=10, headers=PEXELS_AUTHORIZATION).json()
        videos = request['videos']
        print('{}/{}'.format(i, page_numbers))
        for i in tqdm(range(len(videos))):
            video_id = videos[i]['url'].split('/')[-2]
            path = osp.join('datasets', 'nature_video', 'images', video_id)
            for j in range(len(videos[i]['video_files'])):
                if videos[i]['video_files'][j]['quality']=='sd' and videos[i]['video_files'][j]['file_type']=='video/mp4': # SDのmp4形式をダウンロード
                    url_video = videos[i]['video_files'][j]['link']
                    r = requests.get(url_video)
                    with open(path+'.mp4', 'wb') as outfile:
                        outfile.write(r.content)
            cap = cv2.VideoCapture(path+'.mp4')
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) # FPSを取得
            thresh = fps / target_fps
            digit_length = min(len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/thresh - 1))), len(str(max_frames - 1)))  # 連番画像の最大桁数を取得
            n = 0
            counter = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    counter += 1
                    if counter >= thresh:
                        height, width = frame.shape[0], frame.shape[1]
                        min_edge = min(height, width)
                        frame_trim = frame[height//2-min_edge//2:height//2+min_edge//2, width//2-min_edge//2:width//2+min_edge//2] # トリミング
                        frame_trim = cv2.resize(frame_trim, dsize=(target_size, target_size))
                        cv2.imwrite('{}_{}.jpg'.format(path, str(n).zfill(digit_length)), frame_trim)
                        n += 1
                        if n == max_frames:
                            break
                        counter -= thresh
                else:
                    break
            os.remove(path+'.mp4')

    make_vdo_list('datasets/nature_video')


def create_ukiyoe_video():
    if not osp.exists('datasets/ukiyoe'):
        print('ukiyoeデータセットがありません。先にukiyoeデータセットをダウンロードしてください。')
        return
    print('ukiyoeデータセットを作成します')
    dir_path_ukiyoe = osp.join('datasets', 'ukiyoe')
    dir_path_ukiyoe_video = osp.join('datasets', 'ukiyoe_video')
    check_dir(dir_path_ukiyoe_video+'/images')

    digit_length = len(str(NUM_FILTER))-1
    for i, file_path in enumerate(tqdm(os.listdir(dir_path_ukiyoe+'/images'))):
        image = cv2.imread(dir_path_ukiyoe+'/images/'+file_path)
        images = transform_image(image, NUM_FILTER)
        for j, image_ in enumerate(images[1:]):
            cv2.imwrite(dir_path_ukiyoe_video+f'/images/{i}_{str(j).zfill(digit_length)}.jpg', image_)

    make_vdo_list('datasets/ukiyoe_video')

    return


def main():
    parser = argparse.ArgumentParser(description='データセットのダウンロード')
    parser.add_argument('-l', '--list', help='ダウンロードできるデータセットの一覧', action='store_true')
    parser.add_argument('-d', '--datasets', nargs='*', help='ダウンロードするデータセットを選択(空白で複数選択可能)')
    parser.add_argument('--all', help='全てのデータセットをダウンロード', action='store_true')
    args = parser.parse_args() 

    if args.list:
        for dataset in DATASETS:
            print(dataset)

    if args.all:
        download_berkeley_A2B(A='ukiyoe', B='photo')
        download_berkeley_A2B(A='vangogh', B='photo')
        download_nature_video()
        create_ukiyoe_video()
        return

    if args.datasets:
        if 'ukiyoe' in args.datasets:
            download_berkeley_A2B(A='ukiyoe', B='photo')
        if 'vangogh' in args.datasets:
            download_berkeley_A2B(A='vangogh', B='photo')
        if 'nature_video' in args.datasets:
            download_nature_video()
        if 'ukiyoe_video' in args.datasets:
            create_ukiyoe_video()

if __name__ == "__main__":
    main()