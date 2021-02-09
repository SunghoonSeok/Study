import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    validation_split=0.2505
)
test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(56,56),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(56,56),
    batch_size=500,
    class_mode='binary',
    subset='validation'
)
# Found 1301 images belonging to 2 classes.
# Found 434 images belonging to 2 classes.
print(xy_train[0][0].shape) # (14, 56, 56, 3)
print(xy_train[0][1].shape) # (14,)

