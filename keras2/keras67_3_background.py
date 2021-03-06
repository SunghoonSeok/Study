import numpy as np
import PIL
from numpy import asarray
from PIL import Image

import matplotlib.pyplot as plt
import cv2
for i in range(1000,2000) :
    try:

        image_rgb = cv2.imread(f'../data/image/sex/female/final_{i}.jpg')


        # 사각형 좌표: 시작점의 x,y  ,height, weight
        rectangle = (0, 0, 299, 299)

        # 초기 마스크 생성
        mask = np.zeros(image_rgb.shape[:2], np.uint8)

        # grabCut에 사용할 임시 배열 생성
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # grabCut 실행
        cv2.grabCut(image_rgb, # 원본 이미지
                mask,       # 마스크
                rectangle,  # 사각형
                bgdModel,   # 배경을 위한 임시 배열
                fgdModel,   # 전경을 위한 임시 배열 
                2,          # 반복 횟수
                cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화


        # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
        mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

        # 이미지에 새로운 마스크를 곱해 배경을 제외
        image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
        # if i<1000:
        #     cv2.imwrite(f'../data/image/sex//male_mod/image{i}.jpg',image_rgb_nobg)
        # elif i>= 1000 :
        cv2.imwrite(f'../data/image/sex/female_mod/image{i}.jpg',image_rgb_nobg)
    except:
        continue