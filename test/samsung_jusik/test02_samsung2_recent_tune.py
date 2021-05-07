import numpy as np
import pandas as pd

# 1. 데이터
data = np.load('c:/data/test/samsung_data.npy')
x = data[:,:-1]
y = data[:,-1]

print(x.shape, y.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# from sklearn.preprocessing import MaxAbsScaler
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# scaler.fit(x)
# x = scaler.transform(x)


size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)


x = x_data[:-1,:,:]
y = y[6:]
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, GRU, SimpleRNN, Conv1D, Flatten, LeakyReLU
inputs = Input(shape=(6,x.shape[2]))
# dense1 = LSTM(512)(inputs)
# # dense1 = Dense(512)(dense1)
dense1 = LSTM(256)(inputs)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(4)(dense1)
dense1 = Dense(2)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)

#3. 컴파일, 훈련
from tensorflow.keras import optimizers
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=100, mode='auto')
modelpath= 'c:/data/test/tune/samsung3_checkpoint_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=32, epochs=2000, validation_split=0.25, callbacks=[cp, es])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

x_pred = x_data[-8:,:,:]
x_pred = x_pred.reshape(x_pred.shape[0],6,x_pred.shape[-1])
y_predict = model.predict(x_pred)
y_price = int(np.round(y_predict[-1]))

print("이전 값들과 비교")
for i in range(1,(x_pred.shape[0])):
    subset = ([int(y_predict[i-1]),y[-(x_pred.shape[0])+i]])
    print(subset)
print("--------------------------------")
print("익일 삼성 주가 : ", y_price, "원")

# 모델 2
# loss, mae :  1303524.25 838.7462158203125
# RMSE :  1141.7198306777173
# R2 :  0.9751952448930796
# 이전 값들과 비교
# [84694, 82200.0]
# [84967, 82900.0]
# [88074, 88800.0]
# [91000, 91000.0]
# [90766, 90600.0]
# [90169, 89700.0]
# [89948, 89700.0]
# --------------------------------
# 익일 삼성 주가 :  90827 원
