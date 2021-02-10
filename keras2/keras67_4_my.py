# 나를 찍어서 내가 남자인지 여자인지에 대해

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten
import PIL
from numpy import asarray
from PIL import Image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    validation_split=0.2505
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
# Found 1389 images belonging to 2 classes.
# Found 347 images belonging to 2 classes.
# print(xy_train[0][0].shape) # (14, 150, 150, 3)
# print(xy_train[0][1].shape) # (14,)
# print(xy_val[0][0].shape)

# model = Sequential()
# model.add(Conv2D(32, 2, padding='same', activation='relu', input_shape=(150,150,3)))
# model.add(BatchNormalization())
# model.add(Conv2D(64,3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(128,2, activation='relu'))
# model.add(MaxPooling2D(2))
# model.add(Conv2D(64,3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64,3, activation='relu'))
# model.add(MaxPooling2D(2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# from tensorflow.keras.optimizers import Adam
# optimizer = Adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_loss', patience = 30)
# lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)
# filepath = 'c:/data/modelcheckpoint/keras67_1_checkpoint3.hdf5'
# cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
# history = model.fit_generator(xy_train, steps_per_epoch=(xy_train.samples/xy_train.batch_size), epochs=500, validation_data=xy_val, validation_steps=(xy_val.samples/xy_val.batch_size),
# callbacks=[es,cp,lr])
# from tensorflow.keras.models import load_model
# from sklearn.metrics import r2_score


model = load_model('c:/data/modelcheckpoint/keras67_1_checkpoint3.hdf5', compile='False')
loss, acc = model.evaluate_generator(xy_val)

xy_pred = test_datagen.flow_from_directory(
    '../data/image/my_face',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
)


result = model.predict_generator(xy_pred, verbose=True)


print("loss :", loss)
print("acc :", acc)
print('남자일 확률은 ', result[0]*100,'% 입니다.')
# print('남자일 확률은 ', result[1]*100,'% 입니다.')
# print('남자일 확률은 ', result[2]*100,'% 입니다.')
# print('남자일 확률은 ', result[3]*100,'% 입니다.')
# print('남자일 확률은 ', result[4]*100,'% 입니다.')
# print('남자일 확률은 ', result[5]*100,'% 입니다.')
print(np.where(result < 0.5, '여자', '남자'))

# loss : 0.5691425204277039
# acc : 0.7004608511924744
# 남자일 확률은  [57.703728] % 입니다.
# 남자일 확률은  [57.065094] % 입니다.
# [['남자']
#  ['남자']]