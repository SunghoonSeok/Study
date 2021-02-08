import numpy as np
import pandas as pd
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, Xception
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *

dataset = pd.read_csv('c:/data/test/dirty_mnist/dirty_mnist_2nd_answer.csv')
submission = pd.read_csv('c:/data/test/dirty_mnist/sample_submission.csv')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory('c:/data/test/dirty_mnist/dirty_mnist_2nd/*.png', target_size=(256,256), 
                  color_mode='grayscale', class_mode='binary', subset='training')
val_generator = datagen.flow_from_directory('c:/data/test/dirty_mnist/dirty_mnist_2nd/*.png', target_size=(256,256), 
                  color_mode='grayscale', class_mode='binary', subset='validation')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
x_train, y_train = train_generator.next()
print(x_train.shape)
'''
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in alphabet:
    y = dataset.loc[:,alphabet]
    model = InceptionResNetV2(weights=None, include_top=True, input_shape=(256, 256, 1), classes=1)    
    optimizer = Adam(lr=0.002)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', 
    monitor='val_loss', save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(patience=50,verbose=1,factor=0.5) #learning rate scheduler
    es = EarlyStopping(patience=150, verbose=1)
    model.fit_generator(train_generator, epochs=500, validation_data=val_generator, callbacks=[checkpoint,lr,es])
    model2 = load_model(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', compile=False)
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory('c:/data/test/dirty_mnist/test_dirty_mnist_2nd/', target_size=(256,256), 
    color_mode='grayscale', class_mode='binary', shuffle=False)
    predict = model2.predict_generator(test_generator).argmax(axis=1)
    submission[i] = predict
submission.to_csv('../data/test/mnist/submission_ensemble3.csv', index=False)


'''