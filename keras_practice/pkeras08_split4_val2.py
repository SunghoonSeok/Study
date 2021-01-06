'''
train test split만 사용하여 train/val/test 나누기
각각의 갯수 확인
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)




#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=150)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
r2= r2_score(y_test,y_predict)

print("loss : ", loss)
print("mae : ", mae)
print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2)

print(x_train.size)
print(x_val.size)
print(x_test.size)




