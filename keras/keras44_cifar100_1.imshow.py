import numpy as np
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(x_train[0])
print(y_train[0]) 
print(x_train[0].shape) # (32, 32, 3)
print(y_train[0:50]) # 0~99

plt.imshow(x_train[0]) # 소

plt.show() 