import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

batch = 32
seed = 516
filenum = 17
img_size = 192

for i in range(700,1000):
    os.mkdir('../data/lotte/train_new/{0:04}'.format(i))
for i in range(1000):
    for img in range(48):
        image = Image.open(f'../data/lotte/train/{i}/{img}.jpg')
        image.save('../data/lotte/train_new/{0:04}/{1:02}.jpg'.format(i, img))

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=40,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.1,
    preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05,
    preprocessing_function=preprocess_input
)

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/lotte/train_new',
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode='sparse',
    subset='training',
    seed= seed
)
xy_val = train_datagen.flow_from_directory(
    '../data/lotte/train_new',
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode='sparse',
    subset='validation',
    seed=seed
)

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, GaussianDropout
efficientnet = EfficientNetB4(include_top=False,weights='imagenet',input_shape=(img_size,img_size,3))
efficientnet.trainable = True
a = efficientnet.output
a = Dense(2048, activation= 'swish') (a)
a = GaussianDropout(0.3) (a)
a = GlobalAveragePooling2D() (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnet.input, outputs = a)

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.0005)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
file_path = 'c:/data/modelcheckpoint/lotte_last5.hdf5'
mc = ModelCheckpoint(file_path, monitor='val_loss',save_best_only=True,mode='min',verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,verbose=1,mode='min')

model.fit_generator(xy_train, steps_per_epoch=(xy_train.samples/xy_train.batch_size), 
epochs=1000, validation_data=xy_val, validation_steps=(xy_val.samples/xy_val.batch_size),
callbacks=[es,mc,rl])

from tensorflow.keras.models import load_model
model = load_model('c:/data/modelcheckpoint/lotte_last5.hdf5')

for i in range(72000):
    image = Image.open(f'../data/lotte/test/test/{i}.jpg')
    image.save('../data/lotte/test_new/test_new/{0:05}.jpg'.format(i))

test_data = test_datagen.flow_from_directory(
    '../data/lotte/test_new',
    target_size=(img_size,img_size),
    batch_size=batch,
    class_mode=None,
    shuffle=False
)

sub = pd.read_csv('../data/lotte/sample.csv')
cumsum = np.zeros([72000, 1000])
count_result = []
for tta in range(50):
    print(f'{tta+1} 번째 TTA 진행중 - TTA')
    pred = model.predict(test_data, steps=len(test_data), verbose=1) # (72000, 1000)
    pred = np.array(pred)
    cumsum = np.add(cumsum, pred)
    temp = cumsum / (tta+1)
    temp_sub = np.argmax(temp, 1)
    sub.loc[:, 'prediction'] = temp_sub
    sub.to_csv('../data/lotte2/answer ({0:02}).csv'.format((tta+1+11)), index = False)