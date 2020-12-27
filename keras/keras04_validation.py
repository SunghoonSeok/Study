import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([1, 2, 3, 4, 5])

x_validation = np.array([6, 7, 8])
y_validation = np.array([6, 7, 8])

x_test = np.array([9, 10, 11])
y_test = np.array([9, 10, 11])

#2. 모델구성
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # acc라고 줄여도 됨
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=2,
          validation_data=(x_validation, y_validation) #epoch한 번 돌아갈때 val도 같이 한번 돌아감
          )

          

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print('loss : ', loss)

#result = model.predict([9])
result = model.predict(x_train)
print("result : ", result)


# validation_loss보다 loss값이 더 낮다
# val_loss(검증셋)를 더 신뢰함.
# loss는 가장 중요한 지표
# 검증데이터는 훈련데이터에 반영이 된다. 그러니 loss가 더 낮을 수 밖에