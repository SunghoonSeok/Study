#실습
#R2를 음수가 아닌 0.5 이하로 줄이기
#1. 레이어는 인풋과 아웃풋을 포함 6개 이상
#2. batch_size =1
#3. epochs = 100 이상
#4. 데이터 조작 금지.
#5. 히든 레이어의 노드의 갯수는 10 이상
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array
#1. 데이터
x_train = array([1,2,3,4,5,6,7,8,9,10])
y_train = array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18])
#2. 모델구성
model = Sequential()
model.add(Dense(10000, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10000, activation='relu'))
model.add(Dense(10))
model.add(Dense(10000))
model.add(Dense(10))
model.add(Dense(10000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=147, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

# RMSE, R2
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2)






