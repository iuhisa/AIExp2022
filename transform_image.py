import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1)

'''
参考 https://qiita.com/koshian2/items/c133e2e10c261b8646bf
'''


def shift_x(image: np.ndarray, shift_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift_pixel
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shift_x(shift_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += shift_pixel
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


def shift_y(image: np.ndarray, shift_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += shift_pixel
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shift_y(shift_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += shift_pixel
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


def expand(image: np.ndarray, ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_expand(ratio: float) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


#起点 左下
def shear_x_bottom(image: np.ndarray, shear_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear_pixel / h * (h - src[:,1])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shear_x_bottom(h: int, w: int, shear_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear_pixel / h * (h - src[:,1])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)

#起点 左上
def shear_x_top(image: np.ndarray, shear_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear_pixel / h * src[:,1]).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shear_x_top(h: int, w: int, shear_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear_pixel / h * src[:,1]).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


#起点 右上
def shear_y_right(image: np.ndarray, shear_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += (shear_pixel / w * (w - src[:,0])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shear_y_right(h: int, w: int, shear_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += (shear_pixel / w * (w - src[:,0])).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


#起点 左上
def shear_y_left(image: np.ndarray, shear_pixel: int) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += (shear_pixel / w * src[:,0]).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_shear_y_left(h: int, w: int, shear_pixel: int) -> np.ndarray:
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,1] += (shear_pixel / w * src[:,0]).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return np.append(affine, [[0,0,1]], axis=0)


def rotate_center(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    return cv2.warpAffine(image, affine, (w, h))


def get_array_rotate_center(h: int, w: int, angle: float) -> np.ndarray:
    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    return np.append(affine, [[0,0,1]], axis=0)


def transform_image(image: np.ndarray, num_filter:int = 10)->list:

    images = [image]

    h, w = image.shape[:2]
    affine = np.eye(3, dtype = np.float32)

    shift_pixel_x = random.randint(-2, 2)
    shift_pixel_y = random.randint(-2, 2)
    #ratio = random.uniform(-0.016, 0.016)+1 #256x256で最大左右それぞれ2pixel大きくなる
    ratio = 1
    shear_pixel_x = random.randint(-2, 2)
    shear_pixel_y = random.randint(-2, 2)
    angle = random.uniform(-1, 1) #端で大体2pixel動く
    
    array_shift_x = get_array_shift_x(shift_pixel_x)
    array_shift_y = get_array_shift_y(shift_pixel_y)
    array_expand = get_array_expand(ratio)
    array_shear_x = get_array_shear_x_bottom(h, w, shear_pixel_x) if random.random() >= 0.5 else get_array_shear_x_top(h, w, shear_pixel_x)
    array_shear_y = get_array_shear_y_left(h, w, shear_pixel_y) if random.random() >= 0.5 else get_array_shear_y_right(h, w, shear_pixel_y)
    array_rotate_center = get_array_rotate_center(h, w, angle)

    affine = affine.dot(array_shift_x.dot(array_shift_y.dot(array_expand.dot(array_shear_x.dot(array_shear_y.dot(array_rotate_center))))))
    
    for i in range(num_filter-1):
        '''
        image_ = shift_x(images[-1], shift_pixel_x)
        image_ = shift_y(image_, shift_pixel_y)
        image_ = expand(image_, ratio)
        image_ = rotate_center(image_, angle)
        if random.random() >= 0.5:
            image_ = shear_x_top(image_, shear_pixel_x)
        else:
            image_ = shear_x_bottom(image_, shear_pixel_x)
        if random.random() >= 0.5:
            image_ = shear_y_left(image_, shear_pixel_y)
        else:
            image_ = shear_y_right(image_, shear_pixel_y)
        '''
        affine_ = affine.copy()
        print(affine_)
        affine_[0][0]=affine_[0][0]-1
        affine_[1][1]=affine_[1][1]-1
        affine_ = affine_*(i+1)
        affine_[0][0]=affine_[0][0]+1
        affine_[1][1]=affine_[1][1]+1
        print(affine_)
        images.append(cv2.warpAffine(images[0], np.delete(affine_, 2, axis=0), (w, h)))

    return images


def main():
    img = cv2.imread("example.jpg")

    num_filter = 10
    converted = transform_image(img, num_filter)
    for i in range(num_filter-1):
        plt.subplot(250+i+1).imshow(cv2.cvtColor(converted[i+1], cv2.COLOR_BGR2RGB))
    plt.show()

    return 


if __name__ == "__main__":
    main()