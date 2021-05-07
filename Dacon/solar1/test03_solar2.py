# 날짜7일로 날짜2일 묶어서 출력 -> 잘못된 방법
# 총 일수, 묶은 일수, 일당 데이터, 컬럼 4차원 행렬
# conv2d

import numpy as np
import pandas as pd

data = np.load('c:/data/test/solar/train.npy')
print(data.shape)
data = data.reshape(1095, 48, 9)

def split_xy(dataset, timesteps_x, timesteps_y):
    x, y = list(), list()
    
    for i in range(len(data)):
        x_end_number = i + timesteps_x
        y_end_number = x_end_number + timesteps_y
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

timesteps_x = 7
timesteps_y = 2

x, y = split_xy(data, timesteps_x, timesteps_y)
print(x.shape, y.shape) # (1087, 7, 48, 9) (1087, 2, 48, 9)
x = x[:,:,:,3:]
y = y[:,:,:,3:]
print(x.shape, y.shape) # (1087, 7, 48, 6) (1087, 2, 48, 6)

y = y.reshape(1087, 2*48*6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv2D, Reshape, Flatten
inputs = Input(shape=(7,48,6))
dense1 = Conv2D(512, 2, padding='same')(inputs)
dense1 = Conv2D(256,2)(dense1)
dense1 = Conv2D(128,2)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Flatten()(dense1)
outputs = Dense(2*48*6)(dense1)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=80, mode='auto')
modelpath= 'c:/data/test/solar/solar_checkpoint_{val_loss}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr =ReduceLROnPlateau(monitor='val_loss', patience=40, factor=0.5, verbose=1)
model.fit(x_train, y_train, batch_size=64, epochs=500, validation_split=0.2, callbacks=[cp, es, reduce_lr])

model.save('c:/data/test/solar/solar_model.h5')

# 4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
