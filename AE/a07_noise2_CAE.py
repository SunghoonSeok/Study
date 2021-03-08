import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 1.1이 될수도 있으니 clip으로 값을 0~1사이로 고정시킴
x_test_noised = np.clip(x_test_noised, a_min=0,a_max=1)

x_train_noised_in = x_train_noised.reshape(60000, 28, 28, 1)
x_test_noised_in = x_test_noised.reshape(10000, 28, 28, 1)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
#     model.add(Dense(units=784,activation='sigmoid'))
#     return model
def autoencoder():
    model = Sequential()
    # model.add(Dense(units = 256, activation = 'relu', input_shape = (784,)))
    model.add(Conv2D(256, 3, activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(3))
    model.add(Conv2D(256, 5, activation = 'relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(units = 784, activation = 'sigmoid'))
    return model

model = autoencoder()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train_noised,x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = \
    plt.subplots(3,5,figsize=(20,7))

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("NOISE",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()