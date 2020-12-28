# 실습 validation_data 만들기
# train_test_split 사용할것

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
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False)
# 위 두가지 경우에 대해 확인 후 정리할 것
# 넘칠 경우 value error가 뜬다. ValueError: The sum of test_size and train_size = 1.1
# 부족할 경우 56 : 14 : 20이 나온다. shuffle을 안하면 train : 1~56 val : 57~70 test : 71~90 값을 가진다.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, shuffle=True) # 80 : 20
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle=False) # 64 : 16 : 20
 #(x, y, train_size=0.6, shuffle=False) 라고 하면 안섞이고 순서대로 나옴
print(x_test)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
'''
#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) 

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_predict)

# 사이킷런
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# shuffle = False
# loss :  0.002067783148959279
# mae :  0.04490470886230469

# shuffle = True
# loss :  0.0010715597309172153
# mae :  0.02582155540585518

# validation = 0.2 -> 왜 떨어졌지? -> 훈련량이 적어진것이 하나의 원인
# loss :  0.0016458295285701752
# mae :  0.028613686561584473

# train_test_split 활용 val =0.2
# loss :  0.003060322254896164
# mae :  0.0488809272646904
# RMSE :  0.05532017991865325
# R2 :  0.9999963419965022
'''