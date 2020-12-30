# 실습 다:1 앙상블을 구현하시오.

import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])

y = np.array([range(711, 811), range(1, 101), range(201, 301)])


#y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, shuffle=True, train_size=0.8)
# x1, x2를 합칠수 있었다..!
# shuffle =True/False에 따라 validation loss 차이가 극심하게 났다. 왜일까?


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델 2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(50, activation='relu')(dense2)
dense2 = Dense(50, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(100)(middle1)
middle1 = Dense(100)(middle1)
middle1 = Dense(10)(middle1)

# 모델 분기 1
output = Dense(30)(middle1)
output = Dense(70)(output)
output = Dense(3)(output)


 # 모델 선언
model = Model(inputs=[input1, input2], outputs=output) # 2개 이상은 리스트로 묶는다
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train,
           epochs=200, batch_size=4, validation_split=0.3, verbose=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test, batch_size=4)
print("model.metrics_name : ", model.metrics_names)
print("loss : ", loss)

y_predict = model.predict([x1_test, x2_test])
print("====================================")
print("y1_predict : \n", y_predict)
print("====================================")


# RMSE, R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# print("RMSE : ", (RMSE(y1_test, y1_predict)+RMSE(y2_test, y2_predict))/2)
#print("mse : ", mean_squared_error(y1_test, y1_predict))
RMSE = RMSE(y_test, y_predict)

print("RMSE : ", RMSE)

from sklearn.metrics import r2_score
#def R2(y_test, y_predict):
#    return r2_score(y_test, y_predict)

#print("R2 : ", (R2(y1_test, y1_predict)+R2(y2_test, y2_predict))/2)

r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)









