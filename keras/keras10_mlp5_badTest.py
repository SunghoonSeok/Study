#실습
# 1. R2     :    0.5 이하/ 음수 X
# 2. layer  :    5개 이상
# 3. node   :    각 10개 이상
# 4. batch_size  :  8이하
# 5. epochs   : 30 이상
# 다 : 다 mlp
import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(201, 301), range(501, 601)])
y = np.array([range(711, 811), range(1, 101)])
print(x.shape) #(5, 100)
print(y.shape) # (2, 100)


x = np.transpose(x)  # x = x.T
y = np.transpose(y)
print(x) 
print(x.shape)   #(100, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)  
 # 행을 자르는겨, 열(특성)은 건들지않아    random_state

print(x_train.shape) #(80, 5)
print(y_train.shape) #(80, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))  # output_dim = 2

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=60, batch_size=5, validation_split=0.2)

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

# x_predict = np.array([[100,401,101,301,601]])
x_predict = np.array([100,401,101,301,601])
x_predict = x_predict.reshape(1,5)

y_predict = model.predict(x_predict)
print(y_predict)

