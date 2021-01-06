# 엉망인 데이터로 loss값 줄이기

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])
x_pred = np.array([111,112,113])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=3, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=3)
print("loss : ", loss)
y_predict = model.predict(x_pred)
print("y_predict : ", y_predict)










