import pandas as pd
import numpy as np
from pandas import read_csv
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
submission = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)


# train 이미지들과 test 이미지들을 저장해놓을 폴더 생성
import os
# os.mkdir('../data/test/mnist/images_train')
# for i in range(10):
#     os.mkdir(f'../data/test/mnist/images_train/{i}')
# os.mkdir('../data/test/mnist/images_test')
# os.mkdir('../data/test/mnist/images_test/none')

# cv2 이용을 위해서 pip install opencv-python
# 이미지 저장
import cv2

# for idx in range(len(train)) :
#     img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     digit = train.loc[idx, 'digit']
#     cv2.imwrite(f'../data/test/mnist/images_train/{digit}/{train["id"][idx]}.png', img)


# for idx in range(len(test)) :
#     img = test.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     cv2.imwrite(f'../data/test/mnist/images_test/none/{test["id"][idx]}.png', img)

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
'''
model_1 = InceptionResNetV2(weights=None, include_top=True, input_shape=(224, 224, 1), classes=10)

model_2 = Sequential()
model_2.add(InceptionV3(weights=None, include_top=False, input_shape=(224,224,1)))
model_2.add(GlobalAveragePooling2D())
model_2.add(Dense(1024, kernel_initializer='he_normal', activation='relu'))
model_2.add(BatchNormalization())
model_2.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
model_2.add(BatchNormalization())
model_2.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model_2.add(BatchNormalization())
model_2.add(Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions'))


model_3 = Sequential()
model_3.add(Xception(weights=None, include_top=False, input_shape=(224,224,1)))
model_3.add(GlobalAveragePooling2D())
model_3.add(Dense(1024, kernel_initializer='he_normal', activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dense(10, kernel_initializer='he_normal', activation='softmax', name='predictions'))

from tensorflow.keras.optimizers import Adam
optimizer1 = Adam(lr=0.002)
optimizer2 = Adam(lr=0.002)
optimizer3 = Adam(lr=0.002)


model_1.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
model_2.compile(loss='categorical_crossentropy', optimizer=optimizer2, metrics=['accuracy'])
model_3.compile(loss='categorical_crossentropy', optimizer=optimizer3, metrics=['accuracy'])
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                             rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1)
'''
train_generator = datagen.flow_from_directory('../data/test/mnist/images_train', target_size=(224,224), 
                  color_mode='grayscale', class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory('../data/test/mnist/images_train', target_size=(224,224), 
                  color_mode='grayscale', class_mode='categorical', subset='validation')

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint_1 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint7_1.hdf5', 
monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_2 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint7_2.hdf5', 
monitor='val_accuracy', save_best_only=True, verbose=1)
checkpoint_3 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint7_3.hdf5', 
monitor='val_accuracy', save_best_only=True, verbose=1)
lr = ReduceLROnPlateau(patience=50,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=150, verbose=1)

hist1 = model_1.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_1,lr,es])
hist2 = model_2.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_2,lr,es])
hist3 = model_3.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint_3,lr,es])

import matplotlib.pyplot as plt

plt.plot(hist1.history["accuracy"], label='m1_acc')
plt.plot(hist1.history["val_accuracy"], label='m1_vacc')

plt.plot(hist2.history["accuracy"], label='m2_acc')
plt.plot(hist2.history["val_accuracy"], label='m2_vacc')

plt.plot(hist3.history["accuracy"], label='m3_acc')
plt.plot(hist3.history["val_accuracy"], label='m3_acc')

plt.legend()
plt.show()
'''
from tensorflow.keras.models import load_model
model_1 = load_model('c:/data/test/mnist/checkpoint/mnist_checkpoint7_1.hdf5', compile=False)
model_2 = load_model('c:/data/test/mnist/checkpoint/mnist_checkpoint7_2.hdf5', compile=False)
model_3 = load_model('c:/data/test/mnist/checkpoint/mnist_checkpoint7_3.hdf5', compile=False)


# os.mkdir('../data/test/mnist/images_test/none')
# import shutil
# shutil.move("../data/test/mnist/images_test/*.png", "../data/test/mnist/images_test/none")

datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory('../data/test/mnist/images_test', target_size=(224,224), 
color_mode='grayscale', class_mode='categorical', shuffle=False)

predict_1 = model_1.predict_generator(test_generator).argmax(axis=1)
predict_2 = model_2.predict_generator(test_generator).argmax(axis=1)
predict_3 = model_3.predict_generator(test_generator).argmax(axis=1)

submission["predict_1"] = predict_1
submission["predict_2"] = predict_2
submission["predict_3"] = predict_3
print(submission.head())

from collections import Counter

for i in range(len(submission)) :
    predicts = submission.loc[i, ['predict_1','predict_2','predict_3']]
    submission.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]



submission = submission[['id', 'digit']]
temp = submission.copy()
temp1 = temp['digit'].iloc[:7951]
temp2 = temp['digit'].iloc[7951:]
print(temp1.shape,temp2.shape)
print(temp1)
print(temp2)
digit = pd.concat([temp2, temp1])
print(digit)
print(digit.shape)
digit = np.array(digit)
submission.loc[:,'digit'] = digit
print(submission)
submission.to_csv('../data/test/mnist/submission_ensemble3.csv', index=False)
