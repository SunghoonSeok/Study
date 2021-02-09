import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)
# Found 160 images belonging to 2 classes.
# Found 160 images belonging to 2 classes.

print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001D03FBC8550>
print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][0].shape) # (5, 150, 150, 3)
print(xy_train[0][1]) # [1. 1. 0. 1. 1.]
print(xy_train[0][1].shape) # (5,)
print(xy_train[31][1])
print(xy_train[32][1]) # 오류 -> 5장씩 불러오니까 160/5 = 32개의 행이 생긴다.
