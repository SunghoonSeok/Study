'''
슬라이싱 사용하여 train/val/test값 나누기
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]
y_train = y[:60]

x_val = x[60:80]
y_val = y[60:80]

x_test = x[80:]
y_test = y[80:]

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(100))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=200, validation_data=(x_val,y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
r2= r2_score(y_test, y_predict)

print('loss : ', loss)
print('mae : ', mae)
print('RMSE : ', RMSE(y_test, y_predict))
print('R2 : ', r2)













