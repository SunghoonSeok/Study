# keras23_3을 카피해서 lstm층을 두개 만들 것

import numpy as np
# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
             [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3)

# x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3,1), return_sequences=True)) #(행, 열, 몇개씩자르는지)
model.add(LSTM(80, activation='relu')) # 붙여쓰니 더 안좋은데... LSTM이 배출하는 값이 시계열 데이터일까? 연속적인데이터인가? 아님
model.add(Dense(20))
model.add(Dense(
model.add(Dense(20))
model.add(Dense(1))

model.summary() # (None, 3, 20) -> (어떤 데이터개수를 주던 받아들이겠다, return_sequence로 행받기, 전layer의 output(node))
# 4*10(10+10+1)=840

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=2000, batch_size=32, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=32)


x_pred = x_pred.reshape(1,3,1)


y_pred = model.predict(x_pred)
print(y_pred)

# LSTM 한번
# loss : 0.0086
# y_pred :[[80.141754]]  -> batch_size=8, early_stopping적용, relu, 마름모형


# loss: 0.0124
# [[80.06566]] 

# loss: 0.0088
# [[80.05471]]

# LSTM 두번
# loss: 0.0025
# [[80.12084]]

# loss: 0.0047
# [[79.27596]]