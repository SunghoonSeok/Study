import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


#데이터 지정 및 전처리
x = np.load("../../data/npy/P_project_x4.npy",allow_pickle=True)
x_pred = np.load('../../data/npy/test.npy',allow_pickle=True)
y = np.load("../../data/npy/P_project_y4.npy",allow_pickle=True)

x = preprocess_input(x)
x_pred = preprocess_input(x_pred) 

data_gen = ImageDataGenerator(
    width_shift_range=(-1,1),
    height_shift_range=(-1,1), 
    rotation_range=20,
    zoom_range=0.2,
    fill_mode='nearest')
data_gen2 = ImageDataGenerator()

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x,y, 
        train_size = 0.9, shuffle = True, random_state=66)  

train_generator = data_gen.flow(x_train,y_train, batch_size=16, seed = 516)
valid_generator = data_gen2.flow(x_valid,y_valid)


mc = ModelCheckpoint('../data/modelcheckpoint/lotte_projcet3.h5',save_best_only=True, verbose=1)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras import regularizers
# efficientnet = EfficientNetB0(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
efficientnet = EfficientNetB0(include_top=True,weights='imagenet')

efficientnet.trainable = True
a = efficientnet.output
# a = Conv2D(filters = 1000,kernel_size=(12,12), strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(1e-5)) (a)
# a = BatchNormalization() (a)
# a = Activation('swish') (a)
# a = GlobalAveragePooling2D() (a)
# a = Dense(2048, activation= 'swish') (a)
# a = Dropout(0.5) (a)
# a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnet.input, outputs = a)

efficientnet.summary()
'''
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(patience= 15)
lr = ReduceLROnPlateau(patience= 5, factor=0.5)

model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])
learning_history = model.fit_generator(train_generator,epochs=200, steps_per_epoch= len(x_train) / 64,
    validation_data=valid_generator, callbacks=[early_stopping,lr,mc])

# predict
model.load_weights('../data/modelcheckpoint/lotte_projcet3.h5')
result = model.predict(x_pred,verbose=True)


sub = pd.read_csv('../data/lotte2/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('../data/lotte2/answer (5).csv',index=False)
'''