# 다 : 1 mlp
# keras10_mlp2.py를 함수형으로 바꾸시오.

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))
print(x.shape) #(3, 100)
print(y.shape) # (100,)


x = np.transpose(x)  # x = x.T
print(x) 
print(x.shape)   #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  
 # 행을 자르는겨, 열(특성)은 건들지않아    random_state

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(40)(dense2)
outputs = Dense(1)(dense3)
model = Model(inputs = input1, outputs=outputs)




# model = Sequential()
# model.add(Dense(10, input_dim=3))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)
print(y_predict)
print('loss : ', loss)
print('mae : ', mae)

# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


x_predict = np.array([100, 401, 101])
x_predict = x_predict.reshape(1,3)
y_predict = model.predict(x_predict)
print(y_predict)