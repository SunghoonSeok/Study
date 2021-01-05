# keras23_LSTM3_scale을 함수형으로 코딩

import numpy as np
# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
             [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])
x = x.reshape(-1, 3, 1)


# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# model = Sequential()
# model.add(LSTM(20, activation='relu', input_shape=(3,1))) #(행, 열, 몇개씩자르는지)
# model.add(Dense(40))
# model.add(Dense(80))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(1))
input1 = Input(shape=(3,1))
dense1 = LSTM(20, activation='relu')(input1)
dense2 = Dense(40)(dense1)
dense3 = Dense(80)(dense2)
dense3 = Dense(40)(dense3)
dense3 = Dense(20)(dense3)
outputs = Dense(1)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=400, batch_size=8, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=8)


x_pred = x_pred.reshape(1,3,1)
y_pred = model.predict(x_pred)
print(y_pred)


# loss : 0.0086
# y_pred :[[80.141754]]  -> batch_size=8, early_stopping적용, relu, 마름모형


# loss: 0.0124
# [[80.06566]] 

# loss: 0.0088
# [[80.05471]]

# 함수형
# loss: 0.0236
# [[79.094475]]