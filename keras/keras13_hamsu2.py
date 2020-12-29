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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)  
 # 행을 자르는겨, 열(특성)은 건들지않아    random_state

print(x_train.shape) #(80, 5)
print(y_train.shape) #(80, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model  # 함수형 모델
from tensorflow.keras.layers import Dense, Input
# from keras.layers import Dense

input1 = Input(shape=(5,))
aaa = Dense(5, activation='relu')(input1)
aaa = Dense(3)(aaa)
aaa = Dense(4)(aaa)    # 이름 어떻게하든 알아서 dense_1, dense_2 이런식으로 바꿔서 계산함
outputs = Dense(2)(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()


# model = Sequential()
# # model.add(Dense(10, input_dim=1, activation='relu'))
# model.add(Dense(5, input_shape=(1,), activation='relu'))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))
# model.summary()




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=5)
'''
# verbose 0 : 훈련과정 출력 안됨
# verbose 1 : 작대기, s, ms/step, loss, mae, val_loss, val_mae, epoch n/100
# verbose 2 : s, loss, mae, val_loss, val_mae, epoch n/100
# verbose 3이상 : epoch  n/100
# default값은 1이다

'''

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

