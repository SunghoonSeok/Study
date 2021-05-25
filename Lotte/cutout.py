import tensorflow
import torch
import math
from random import *
import numpy as np

import cv2
image3 = cv2.imread('c:/users/ai/desktop/20.jpg')
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

def apply_mask(image, size=80, n_squares=1):
    h, w, channels = image.shape
    new_image = image
    for _ in range(n_squares):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_image[30:110,80:160,:] = 0
    return new_image

import matplotlib.pyplot as plt
print("Original images:")

plt.subplot(330 + 1)
plt.imshow(image3)
plt.show()
print("Images with cutout:")

plt.subplot(330 + 1)
plt.imshow(apply_mask(image3))
plt.show()