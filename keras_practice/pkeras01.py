'''
데이터, 모델, 컴파일, 평가 구성 짜보기
loss값, predict값 출력
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([1,2,3])
y = np.array([4,5,6])

#2. 모델 구성

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=200)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)

x_predict = np.array([4,5,6])
y_predict = model.predict(x_predict)
print ("y_predict : ", y_predict)






