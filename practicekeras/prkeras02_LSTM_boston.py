import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)
print("==================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x)) # 711.0  0.0
print(dataset.feature_names)
# 데이터를 0~1 사이로 바꾼다 -> 스케일링 normalization
# # print(dataset.DESCR)

# 데이터 전처리 (MinMax) 전처리는 옵션이아니라 필수
# x = x/711.   -> 모든 열을 하나의 max로 나누는게 맞아? 아니잖아

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x= scaler.transform(x)   # 이렇게 하면 최소값 최대값 알 필요가 없다.
# # 데이터 전체를 fit하게 되면 train의 범위가 0~1이 아니다 val과 test값을 나눠가지니까

# print(np.min(x), np.max(x))
# print(np.max(x[0]))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(13,1))
dense1 = LSTM(64)(inputs)
dense1 = Dense(128)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 전처리 전
# loss, mae :  19.518239974975586 3.345139503479004
# RMSE :  4.4179451574786555
# R2 :  0.7664805756106845

# 전처리 후
# loss, mae :  10.753201484680176 2.6606719493865967
# RMSE :  3.2792074750333793
# R2 :  0.8713469278828012

# MinMax 전처리 후
# loss, mae :  10.743719100952148 2.4805612564086914
# RMSE :  3.277761406247079
# R2 :  0.8714603700318169

# 제대로 전처리 후
# loss, mae :  9.291711807250977 2.191848039627075
# RMSE :  3.0482309493222117
# R2 :  0.8888324313906211

# parameter조정값 -> layer와 node를 줄이고 batch_size와 epochs도 함께 줄임
# loss, mae :  6.998237133026123 1.869640827178955
# RMSE :  2.645418213195852
# R2 :  0.9162719360420687

# early stopping 적용후
# loss, mae :  6.493799686431885 2.0575599670410156
# RMSE :  2.5482935370353137
# R2 :  0.9223071100610187

# LSTM -> relu
# loss, mae :  21.809192657470703 4.053664684295654
# RMSE :  4.670031438221926
# R2 :  0.739071222785505

# LSTM -> default
# loss, mae :  8.408567428588867 2.182143211364746
# RMSE :  2.8997532063408515
# R2 :  0.8993985025399989