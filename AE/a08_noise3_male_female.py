# [실습] keras67_1 남자 여자에 noise를 넣어서
# 기미 주근깨 여드름을 제거하시오
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest',
    validation_split=0.25
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/sex',
    target_size=(150,150),
    batch_size=2000,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex',
    target_size=(150,150),
    batch_size=1000,
    class_mode='binary',
    subset='validation'
)
# Found 1389 images belonging to 2 classes.
# Found 347 images belonging to 2 classes.
print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)
# test_generator

np.save('../data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/npy/keras66_val_x.npy', arr=xy_val[0][0])
np.save('../data/image/brain/npy/keras66_val_y.npy', arr=xy_val[0][1])

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_test = np.load('../data/image/brain/npy/keras66_val_x.npy')

y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_test = np.load('../data/image/brain/npy/keras66_val_y.npy')

x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 1.1이 될수도 있으니 clip으로 값을 0~1사이로 고정시킴
x_test_noised = np.clip(x_test_noised, a_min=0,a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

print(x_train.shape, x_test.shape) # (1301, 150, 150, 3) (434, 150, 150, 3)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
#     model.add(Dense(units=784,activation='sigmoid'))
#     return model
# def autoencoder():
#     model = Sequential()
#     # model.add(Dense(units = 256, activation = 'relu', input_shape = (784,)))
#     model.add(Conv2D(256, 3, activation = 'relu', padding = 'same', input_shape = (150, 150, 3)))
#     model.add(MaxPooling2D(3))
#     model.add(Conv2D(256, 5, activation = 'relu', padding = 'same'))
#     model.add(Flatten())
#     model.add(Dense(128, activation = 'relu'))
#     model.add(Dense(64, activation = 'relu'))
#     model.add(Dense(128, activation = 'relu'))
#     model.add(Dense(256, activation = 'relu'))
#     model.add(Dense(units = 784, activation = 'sigmoid'))
#     return model
def autoencoder():
    model = Sequential()
    model.add(Conv2D(256,3, activation='relu',padding='same',input_shape=(150,150,3)))
    model.add(Conv2D(128,3, activation='relu',padding='same'))
    model.add(Conv2D(64,3, activation='relu',padding='same'))
    model.add(Conv2D(3, 3, padding='same',activation='sigmoid'))
    return model



model = autoencoder()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train_noised,x_train, epochs=30)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = \
    plt.subplots(3,5,figsize=(20,7))

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다!!
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]])
    if i==0:
        ax.set_ylabel("INPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]])
    if i==0:
        ax.set_ylabel("NOISE",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(output[random_images[i]])
    if i==0:
        ax.set_ylabel("OUTPUT",size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()