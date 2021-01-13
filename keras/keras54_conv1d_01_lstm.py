import numpy as np
# 1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
             [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3)

x = x.reshape(13, 3, 1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(20, 4, padding='same', activation='relu', input_shape=(3,1))) #(행, 열, 몇개씩자르는지)
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

model.summary() 
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x, y, epochs=2000, batch_size=16, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=16)


x_pred = x_pred.reshape(1,3,1)


y_pred = model.predict(x_pred)
print(y_pred)


# loss : 0.0086
# y_pred :[[80.141754]]  -> batch_size=8, early_stopping적용, relu, 마름모형


# loss: 0.0124
# [[80.06566]] 

# loss: 0.0088
# [[80.05471]]

# DNN  -> epochs 1000, early_stopping 적용
# loss: 3.2491e-04
# [[80.03067]]

# loss: 0.0013
# [[79.958496]]

# loss: 3.2594e-04
# [[80.03142]]

# LSTM -> epochs=2000, batch32
# loss: 0.0024
# [[79.99867]]

# loss: 3.3963e-04 -> 전처리 후
# [[80.06624]]

# loss: 8.7428e-04
# [[80.14757]]

# loss: 3.2777e-11
# [[80.00001]]
'''