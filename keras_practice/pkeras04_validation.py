'''
validation data 설정하기
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
from numpy import array
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_val = array([6,7,8])
y_val = array([6,7,8])
x_test = array([9,10,11])
y_test = array([9,10,11])
x_pred = array([12])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, batch_size=1, epochs=200, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae)
y_pred = model.predict(x_pred)
print('y_pred : ', y_pred)










