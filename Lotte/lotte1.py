import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator()

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/lotte/train',
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/lotte/train',
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)
# Found 39000 images belonging to 1000 classes.
# Found 9000 images belonging to 1000 classes.
print(xy_train[0][0].shape) # (16, 128, 128, 3)
print(xy_train[0][1].shape) # (16,)


from tensorflow.keras.applications import EfficientNetB0

effi = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128,128,3))
effi.trainable = False

model = Sequential()
model.add(effi)
model.add(Flatten())
model.add(Dense(1000, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=21, mode='min')
file_path = 'c:/data/modelcheckpoint/lotte_efficientnetb0.hdf5'
mc = ModelCheckpoint(file_path, monitor='val_loss',save_best_only=True,mode='min',verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=7,verbose=1,mode='min')

history = model.fit_generator(xy_train, steps_per_epoch=(xy_train.samples/xy_train.batch_size), epochs=100, validation_data=xy_val, validation_steps=(xy_val.samples/xy_val.batch_size),
callbacks=[es,mc,rl])
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score


model = load_model('c:/data/modelcheckpoint/lotte_efficientnetb0.hdf5')
loss, acc = model.evaluate_generator(xy_val)

xy_pred = test_datagen.flow_from_directory(
    '../data/lotte/test',
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)


result = model.predict_generator(xy_pred, verbose=True)

import pandas as pd
submission = pd.read_csv('c:/data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)