import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
train_datagen = ImageDataGenerator(
    zoom_range= [0.8,1.2],
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/lotte/train',
    target_size=(128,128),
    batch_size=16,
    class_mode='sparse',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/lotte/train',
    target_size=(128,128),
    batch_size=16,
    class_mode='sparse',
    subset='validation'
)
# Found 39000 images belonging to 1000 classes.
# Found 9000 images belonging to 1000 classes.
print(xy_train[0][0].shape) # (16, 128, 128, 3)
print(xy_train[0][1].shape) # (16,)




effi = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(128,128,3))
effi.trainable = False

# model = Sequential()
# model.add(effi)
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1000, activation='softmax'))

# model.summary()

# from tensorflow.keras.optimizers import Adam
# optimizer = Adam(lr=0.001)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
# file_path = 'c:/data/modelcheckpoint/lotte_efficientnetb4_2.hdf5'
# mc = ModelCheckpoint(file_path, monitor='val_loss',save_best_only=True,mode='min',verbose=1)
# rl = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,verbose=1,mode='min')

# history = model.fit_generator(xy_train, steps_per_epoch=(xy_train.samples/xy_train.batch_size), epochs=300, validation_data=xy_val, validation_steps=(xy_val.samples/xy_val.batch_size),
# callbacks=[es,mc,rl])
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
model = load_model('c:/data/modelcheckpoint/lotte_efficientnetb4_2.hdf5', compile=False)
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
loss, acc = model.evaluate_generator(xy_val)
print(loss, acc)

x_pred = np.load('../../data/npy/test.npy',allow_pickle=True)
x_pred = preprocess_input(x_pred)

result = model.predict(x_pred, verbose=True)

import pandas as pd
submission = pd.read_csv('c:/data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)

submission.to_csv('c:/data/lotte/sample_efficientb4_2.csv', index=False)