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



model = load_model('c:/data/test/samsung_model.h5')  # 제일 잘 나오는 모델

# model = load_model('c:/data/test/samsung3_checkpoint_2.hdf5')

# model = load_model('c:/data/test/samsung3_model.h5')

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
# loss, mae :  1010360.5 729.6527099609375
# RMSE :  1005.1668662647304
# R2 :  0.9850373516274966
# [[89429.48]]
# 익일 삼성 주가 :  89429 원  
# [[83793.31 ]  83900
#  [81961.82 ]  82200
#  [83425.086]  82900
#  [88169.164]  88800
#  [90805.984]  91000
#  [91019.36 ]  90600
#  [90096.3  ]  89700
#  [89429.49 ]]

# 체크포인트
# loss, mae :  768879.25 675.6148071289062
# RMSE :  876.8574945244251
# R2 :  0.9892928020686379
# [[89346.195]]
# 익일 삼성 주가 :  89346 원
# [[81483.39 ] 83900
#  [80841.766] 82200
#  [83013.27 ] 82900
#  [89141.   ] 88800
#  [90992.34 ] 91000
#  [90229.1  ] 90600
#  [89698.36 ] 89700
#  [89346.195]]