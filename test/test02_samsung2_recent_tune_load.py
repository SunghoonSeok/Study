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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


#2. 모델구성 및 컴파일
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM



# model = load_model('c:/data/test/samsung_model.h5') 

model = load_model('c:/data/test/tune/samsung3_checkpoint_7_1040128.5000.hdf5')

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

# 1차 batch32
# loss, mae :  663316.5 613.0302124023438
# RMSE :  814.4424305452372
# R2 :  0.9897592914903134
# [82412, 82200.0]
# [82640, 82900.0]
# [87062, 88800.0]
# [90756, 91000.0]
# [90400, 90600.0]
# [89669, 89700.0]
# [89790, 89700.0]
# 익일 삼성 주가 :  90547 원

# 2차 GRU, node 1024 ~ 1
# loss, mae :  1165629.25 755.9816284179688
# RMSE :  1079.6427087728032
# R2 :  0.9832344529700237
# [82091, 82200.0]
# [87312, 82900.0]
# [87931, 88800.0]
# [88699, 91000.0]
# [89767, 90600.0]
# [89488, 89700.0]
# [89417, 89700.0]
# 익일 삼성 주가 :  89370 원

# LSTM 1024~1
# loss, mae :  902311.0625 738.9931640625
# RMSE :  949.9005281334481
# R2 :  0.9883271253464256
# [82664, 82200.0]
# [82861, 82900.0]
# [85004, 88800.0]
# [88732, 91000.0]
# [87724, 90600.0]
# [87808, 89700.0]
# [89016, 89700.0]
# 익일 삼성 주가 :  89694 원