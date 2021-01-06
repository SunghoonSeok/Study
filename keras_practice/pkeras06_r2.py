import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

# RMSE, R2
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print(RMSE(y_test,y_predict))

r2 = r2_score(y_test, y_predict)
print(r2)












