'''
サングラスを掛けていない画像を指定すると、その画像をサングラスを掛けた画像に変換する
'''
import dlib
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
glass = cv2.imread("sunglass.png", -1)
width = 1000
height = 334
center_glasses = np.array([[230, 147], [760, 147]])
length_glasses = 760 - 230
center = np.array([500, 167])

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_models/shape_predictor_68_face_landmarks.dat")

img_paths = glob('fake_glass/*.jpg')
# ここで顔を認識してサングラスを掛けてやりたい顔画像をパスで指定

for path in tqdm(img_paths):
    img = cv2.imread(path)
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_empty = np.zeros((img.shape[0], img.shape[1], 4))
    
    faces = detector(img)
    """ print(f'detected faces: {len(faces)}') """
    if len(faces) == 0:
        os.remove(path)
    for face in faces:
        try: 
            shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in shape.parts()])
            left_0 = [shape_2d[36][0], min(shape_2d[37][1], shape_2d[38][1])]
            left_1 = [shape_2d[39][0], max(shape_2d[40][1], shape_2d[41][1])]
            right_0 = [shape_2d[42][0], min(shape_2d[43][1], shape_2d[44][1])]
            right_1 = [shape_2d[45][0], max(shape_2d[46][1], shape_2d[47][1])]
            """ left_eye = img[left_0[1]:left_1[1], left_0[0]:left_1[0]]
            right_eye = img[right_0[1]:right_1[1], right_0[0]:right_1[0]] """
            left_eye_center = np.array([(left_0[0]+left_1[0])//2, (left_0[1]+left_1[1])//2])
            right_eye_center = np.array([(right_0[0]+right_1[0])//2, (right_0[1]+right_1[1])//2])
            eyes_center = (left_eye_center+right_eye_center) / 2
            vector = right_eye_center - left_eye_center
            length_vector = np.linalg.norm(vector, ord=2)
            center_resized = center * (length_vector/length_glasses)
            glass_resized = cv2.resize(glass, dsize=None, fx=length_vector/length_glasses, fy=length_vector/length_glasses)
            glass_resized_width = glass_resized.shape[1]
            glass_resized_height = glass_resized.shape[0]
            last_vector = eyes_center - center_resized
            last_vector_int = np.asarray(last_vector, dtype=int)
            img_empty[last_vector_int[1]:glass_resized_height+last_vector_int[1], last_vector_int[0]:glass_resized_width+last_vector_int[0], :] = glass_resized
            cos_vector = vector[0] / np.linalg.norm(vector, ord=2)
            theta_vector = np.arccos(cos_vector)*180/np.pi
            matrix_angle = theta_vector
            matrix = cv2.getRotationMatrix2D(tuple(eyes_center), matrix_angle, 1)
            img_empty = cv2.warpAffine(img_empty, matrix, (img_width, img_height))
            mask = img_empty[:,:,3]
            mask = mask.astype(np.float32)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask /= 255
            img_float = img.astype(np.float)
            img_float *= 1-mask
            img_float += img_empty[:, :, :3]*mask
            img_int = img_float.astype(np.int)
            cv2.imwrite(path, img_int)
        except:
            os.remove(path)