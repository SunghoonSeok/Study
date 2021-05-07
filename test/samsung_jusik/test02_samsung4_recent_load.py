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
from tensorflow.keras.layers import Dense, Input, LSTM



model = load_model('c:/data/test/samsung_model.h5') 

# model = load_model('c:/data/test/samsung3_checkpoint_3.hdf5')

# model = load_model('c:/data/test/samsung3_model_2.h5')

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

x_pred = x_data[-8:,:,:]
x_pred = x_pred.reshape(x_pred.shape[0],6,x_pred.shape[-1])
y_predict = model.predict(x_pred)
y_price = int(np.round(y_predict[-1]))

for i in range(1,(x_pred.shape[0])):
    subset = ([int(y_predict[i-1]),y[-(x_pred.shape[0])+i]])
    print(subset)


print("익일 삼성 주가 : ", y_price, "원")


# 첫날 주가
# loss, mae :  1357250.625 908.9234008789062
# RMSE :  1165.010832079855
# R2 :  0.9816344505215461
# [[89429.48]]

# 로드모델
# loss, mae :  1098115.5 810.7371215820312
# RMSE :  1047.9098125496105
# R2 :  0.9862533626480259
# [81961, 82200.0]
# [83425, 82900.0]
# [88169, 88800.0]
# [90805, 91000.0]
# [91019, 90600.0]
# [90096, 89700.0]
# [89310, 89700.0]
# 익일 삼성 주가 :  89855 원

# 체크포인트
# loss, mae :  679903.4375 649.822265625
# RMSE :  824.5626122260379
# R2 :  0.9907982638117303
# [83507, 82200.0]
# [84732, 82900.0]
# [87387, 88800.0]
# [90484, 91000.0]
# [91058, 90600.0]
# [90332, 89700.0]
# [89852, 89700.0]
# 익일 삼성 주가 :  89963 원

# 모델2
# loss, mae :  741993.8125 639.5009765625
# RMSE :  861.3906338492835
# R2 :  0.9918425418682768
# [84694, 82200.0]
# [84967, 82900.0]
# [88074, 88800.0]
# [91000, 91000.0]
# [90766, 90600.0]
# [90169, 89700.0]
# [89948, 89700.0]
# 익일 삼성 주가 :  90827 원

# check 3
# loss, mae :  736000.4375 673.4077758789062
# RMSE :  857.9046540225801
# R2 :  0.989360245494436
# [82063, 82200.0]
# [82589, 82900.0]
# [87916, 88800.0]
# [90193, 91000.0]
# [90464, 90600.0]
# [89339, 89700.0]
# [89923, 89700.0]
# 익일 삼성 주가 :  90017 원