import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   
print(x_train[0])
print(y_train[0]) 
print(x_train[0].shape) #(32,32,3)
print(y_train[0:50]) #0~9

plt.imshow(x_train[0]) # 개구리

plt.show() 