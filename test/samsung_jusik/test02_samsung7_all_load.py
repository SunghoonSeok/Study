import numpy as np
import pandas as pd

#1. 데이터
data = np.load('c:/data/test/samsung_jusik2.npy')
x = data[:,:-1]
y = data[:,-1]

print(x.shape, y.shape) 

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

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 


#2. 모델구성 및 컴파일
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten

model = load_model('c:/data/test/samsung6_checkpoint.hdf5')

# model = load_model('c:/data/test/samsung6_model.h5')


#3. 평가, 예측
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

x1_pred = x1_data[-2:,:,:]
x2_pred = x2_data[-2:,:,:]
x1_pred = x1_pred.reshape(x1_pred.shape[0],6,x1_pred.shape[-1])
x2_pred = x2_pred.reshape(x2_pred.shape[0],6,x2_pred.shape[-1])
y_predict = model.predict([x1_pred,x2_pred])

print(y_predict)


print("월요일 삼성 시가 : ", int(np.round(y_predict[0])), "원")
print("화요일 삼성 시가 : ", int(np.round(y_predict[1])), "원")


# loss, mae :  1428569.375 895.7993774414062
# RMSE :  1195.2277438453937
# R2 :  0.9747082894609304
# [[89665.516]]

# 로드모델
# loss, mae :  855456.5625 642.3980712890625
# RMSE :  924.9085310904956
# R2 :  0.9949458441779632
# [[91100.04]]
# 익일 삼성 주가 :  91100 원

# checkpoint
# loss, mae :  791791.875 563.2197875976562
# RMSE :  889.8269283543508
# R2 :  0.9953219792309843
# [[90480.94]]
# 익일 삼성 주가 :  90481 원  
