import numpy as np
import pandas as pd

# 1. 데이터
data = np.load('c:/data/test/samsung_data.npy')
x = data[:,:-1]
y = data[:,-1]

print(x.shape, y.shape) # (662, 14) (662,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)
print(x.shape)

x = x_data[:-1,:,:]
y = y[6:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

x_pred = x_data[-1,:,:]
x_pred = x_pred.reshape(1,6,14)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_pred.shape)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(6,14))
dense1 = LSTM(1000)(inputs)
dense1 = Dense(500)(dense1)
dense1 = Dense(400)(dense1)
dense1 = Dense(300)(dense1)
dense1 = Dense(200)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(10)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='loss', patience=60, mode='auto')
modelpath= 'c:/data/test/samsung3_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_split=0.2, callbacks=[cp])

model.save('c:/data/test/samsung3_model.h5')

#4. 평가, 예측
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

y_predict = model.predict(x_pred)
print(y_predict)

# loss, mae :  1972171.875 1097.6455078125
# RMSE :  1404.3402901598788
# R2 :  0.9730702480193972
# [[91408.875]]

# loss, mae :  1357250.625 908.9234008789062
# RMSE :  1165.010832079855
# R2 :  0.9816344505215461
# [[89429.48]]

# loss, mae :  1346956.75 878.3052978515625
# RMSE :  1160.584646124029
# R2 :  0.9763041431702614
# [[91939.41]]