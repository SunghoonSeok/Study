'''
컴파일에서 metrics를 mae로 설정하기
mae출력하기
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, batch_size=1, epochs=200)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)
y_pred = model.predict(x_train)
print('y_predict : ', y_pred)





