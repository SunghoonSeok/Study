# Simple RNN

# 1. 데이터
import numpy as np


x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
print(x.shape, y.shape) #(4,3) (4,)

#x = x.reshape(4, 3, 1) # LSTM에 넣기 위해서 3차원으로 변환, Dense는 2차원 참고로 CNN은 4차원
x = x.reshape(4, 3, 1)# -1은 제일끝, -2는 끝에서 두번째를 의미 (-1, 3, 1)   (4, -2, 1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1), reset_after=False)) 
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary() 
# param# 390 -> reset after=True
# paran# 360 -> reset after=False



# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)

x_pred = np.array([5,6,7]) # (3,)
x_pred = x_pred.reshape(1,3,1) 
y_pred = model.predict(x_pred)
print(y_pred)

# SimpleRNN
# loss: 0.0012
# [[8.006236]]

# LSTM
# loss: 0.0028
# [[7.9007807]]

# GRU
# loss: 0.0057
# [[8.1311245]]
