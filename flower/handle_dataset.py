import cv2
import glob
import numpy as np
import random
from PIL import Image
import math
from tqdm import tqdm

WIDTH_BACKGROUND = 256
HEIGHT_BACKGROUND = 256
WIDTH_IMAGE = 80
HEIGHT_IMAGE = 80
NUM_TRAIN = 10000 #trainデータセットのサイズ
NUM_TEST = 1000 #testデータセットのサイズ
MIN_IMAGES = 20 #1枚当たりの画像に最大何枚の写真を入れるか
MAX_IMAGES = 30 #1枚当たりの画像に最大何枚の写真を入れるか

def create_color_noise(w:int = WIDTH_BACKGROUND, h:int = HEIGHT_BACKGROUND)->np.ndarray:
    #一様分布
    response = np.random.randint(0, 255, (w, h, 3), dtype=np.uint8) 
    #ガウス分布, 平均,標準偏差
    #response = np.random.normal(0, 2, (w, h, 3))
    '''
    for c in range(3):
        for i in range(w):
            for j in range(h):
                response[i][j] = min(response[i][j][c], 255)'''
    #response = response.astype(np.uint8)
    return response

# 画像のオーバーレイ
def overlay_image(src, overlay, location):
    overlay_height, overlay_width = overlay.shape[:2]

    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    # オーバーレイをPIL形式に変換
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

def main(): #ヒマワリ→たんぽぽ
    urls_sunflower = glob.glob('./dataset/sunflower/*')
    urls_dandelion = glob.glob('./dataset/dandelion/*')
    imgs_sunflower = [cv2.resize(cv2.imread(url), dsize=(WIDTH_IMAGE, HEIGHT_IMAGE)) for url in urls_sunflower]
    imgs_dandelion = [cv2.resize(cv2.imread(url), dsize=(WIDTH_IMAGE, HEIGHT_IMAGE)) for url in urls_dandelion]
    alpha = np.zeros((HEIGHT_IMAGE, WIDTH_IMAGE,1), dtype=np.uint8)
    for i in range(HEIGHT_IMAGE):
        for j in range(WIDTH_IMAGE):
            x = j - WIDTH_IMAGE/2
            y = i - HEIGHT_IMAGE/2
            r = math.sqrt(x**2+y**2)
            sigma = 30 #12
            alpha[i][j] = math.floor(255*math.exp(-(r)**2 / (2*sigma**2)))
    imgs_sunflower = [np.append(img, alpha, axis=2) for img in imgs_sunflower]
    imgs_dandelion = [np.append(img, alpha, axis=2) for img in imgs_dandelion]

    #Trainデータセットの作成
    for i in tqdm(range(NUM_TRAIN)):
        res_sunflower = create_color_noise()
        res_dandelion = create_color_noise()
        max_images = random.randint(MIN_IMAGES, MAX_IMAGES)
        for j in range(max_images):
            img_sunflower = random.choice(imgs_sunflower)
            img_dandelion = random.choice(imgs_dandelion)
            scale = random.randint(3,10)/10
            x = random.randint(1, WIDTH_BACKGROUND-math.floor(WIDTH_IMAGE*scale)-1)
            y = random.randint(1, HEIGHT_BACKGROUND-math.floor(HEIGHT_IMAGE*scale)-1)
            res_sunflower = overlay_image(res_sunflower, cv2.resize(img_sunflower, dsize=(math.floor(WIDTH_IMAGE*scale), math.floor(HEIGHT_IMAGE*scale))), (x,y))
            res_dandelion = overlay_image(res_dandelion, cv2.resize(img_dandelion, dsize=(math.floor(WIDTH_IMAGE*scale), math.floor(HEIGHT_IMAGE*scale))), (x,y))
        cv2.imwrite(f'./dataset/train/src/{i}.jpg', res_sunflower)
        cv2.imwrite(f'./dataset/train/dst/{i}.jpg', res_dandelion)

    #Testデータセットの作成
    for i in tqdm(range(NUM_TEST)):
        res_sunflower = create_color_noise()
        res_dandelion = create_color_noise()
        max_images = random.randint(MIN_IMAGES, MAX_IMAGES)
        for j in range(max_images):
            img_sunflower = random.choice(imgs_sunflower)
            img_dandelion = random.choice(imgs_dandelion)
            scale = random.randint(3,10)/10
            x = random.randint(1, WIDTH_BACKGROUND-math.floor(WIDTH_IMAGE*scale)-1)
            y = random.randint(1, HEIGHT_BACKGROUND-math.floor(HEIGHT_IMAGE*scale)-1)
            res_sunflower = overlay_image(res_sunflower, cv2.resize(img_sunflower, dsize=(math.floor(WIDTH_IMAGE*scale), math.floor(HEIGHT_IMAGE*scale))), (x,y))
            res_dandelion = overlay_image(res_dandelion, cv2.resize(img_dandelion, dsize=(math.floor(WIDTH_IMAGE*scale), math.floor(HEIGHT_IMAGE*scale))), (x,y))
        cv2.imwrite(f'./dataset/test/src/{i}.jpg', res_sunflower)
        cv2.imwrite(f'./dataset/test/dst/{i}.jpg', res_dandelion)

if __name__ == "__main__":
    main()