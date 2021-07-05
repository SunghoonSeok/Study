import numpy
import os
import cv2

list = os.listdir('c:/data/ai_comp_data/task02_train/masks')
a = []
for i in list:
    sum = os.path.splitext(i)
    a.append(sum[0])


for i in a:
    rgb_image = cv2.imread('c:/data/ai_comp_data/task02_train/images/%s'%i)
    cv2.imwrite('c:/data/ai_comp_data/task02_train/new_train/%s'%i,rgb_image)