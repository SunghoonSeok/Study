import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential() # 순차적인
model.add(Dense(100, input_dim=1, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
# model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(learning_rate=0.1))
model.fit(x, y, epochs=137, batch_size=1 )

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)

x_pred = np.array([4])

result = model.predict(x_pred)
print("result : ", result)
