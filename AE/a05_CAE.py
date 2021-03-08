# 4번 카피해서 복붙
# CNN으로 딥하게 구성
# 2개의 모델을 만드는데 하나는 원칙적 오토인코더
# 다른거는 히든 구성
# 2개의 성능 비교

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train1 = x_train.reshape(60000,28,28,1)/255.
x_test1 = x_test.reshape(10000,28,28,1)/255.

x_train2 = x_train.reshape(60000, 784).astype('float32')/255
x_test2 = x_test.reshape(10000, 784).astype('float32')/255

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

def autoencoder():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_1 = x

    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_2 = x




    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x

    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_1)
    x = Dropout(0.4)(x)
    x = Activation('sigmoid')(x)
    x = x
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)


    return model
def deeplearning():
    model = Sequential()
    model.add(Conv2D(256,2,input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128,2,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64,2,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128,2,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(256,2,padding='same'))
    # model.add(Dense(units=(28,28,1),activation='sigmoid'))
    return model

model = autoencoder()
# model2 = deeplearning()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train1, x_train1, epochs=10)
output = model.predict(x_test1)

import matplotlib.pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = \
    plt.subplots(2,5,figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토 인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()