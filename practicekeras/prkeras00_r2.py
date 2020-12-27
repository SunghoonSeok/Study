# 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = array(range(1, 101))
y = array(range(2, 201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:120:2]
y_val = y[120:160:2]
y_test = y[160::2]

#2. 모델 구성
model = Sequential() # 층을 구성하기 위한 코드
model.add(Dense(10, input_dim=1, activation='relu'))
# input_dim은 입력뉴런 개수라고 생각하면 됨
# Dense는 첫번째 hiddenlayer, 추가 dense가 없다면 output이 된다
model.add(Dense(16))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", results)

x_predict = array(range(101, 104))
y_predict = model.predict(x_predict)


print(y_predict)







