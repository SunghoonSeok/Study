from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

# x_train = x[:60] #처음부터 60개 까지 생략된 숫지는 0
# x_val = x[60:80] # 61~80
# x_test = x[80:] # 81~100
# # 리스트의 슬라이싱

# y_train = y[:60] #처음부터 60개 까지 생략된 숫지는 0
# y_val = y[60:80] # 61~80
# y_test = y[80:] # 81~100
# # 리스트의 슬라이싱

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
 #(x, y, train_size=0.6, shuffle=False) 라고 하면 안섞이고 순서대로 나옴
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_predict)

# shuffle = False
# loss :  0.002067783148959279
# mae :  0.04490470886230469

# shuffle = True
# loss :  0.0010715597309172153
# mae :  0.02582155540585518

# validation = 0.2 -> 왜 떨어졌지? -> 훈련량이 적어진것이 하나의 원인
# loss :  0.0016458295285701752
# mae :  0.028613686561584473