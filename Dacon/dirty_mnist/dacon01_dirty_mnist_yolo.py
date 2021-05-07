import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path = 'c:/data/test/dirty_mnist/dirty_mnist_clean/00020.png'

image = cv2.imread(file_path)

edged = cv2.Canny(image, 10, 250)
cv2.imshow('Edged',edged)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closed',closed)
cv2.waitKey(0)

contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total=0

contours_image = cv2.drawContours(image, contours, -1, (0,255,0),3)
cv2.imshow('contours_image',contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours_xy = np.array(contours)
print(contours_xy.shape)
print(contours_xy[0])

x_min, x_max = 0,0
y_min, y_max = 0,0

for i in range(len(contours_xy)):
    value_x = list()
    value_y = list()
    for j in range(len(contours_xy[i])):
        value_x.append(contours_xy[i][j][0][0])
        x_min = min(value_x)
        x_max = max(value_x)
    for j in range(len(contours_xy[i])):
        value_y.append(contours_xy[i][j][0][1])
        y_min = min(value_y)
        y_max = max(value_y)
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    img_trim = image[y:y+h, x:x+w]
    cv2.imshow('image',img_trim)
    cv2.waitKey(0)
