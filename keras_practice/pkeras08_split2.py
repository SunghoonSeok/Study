'''
train_test_split 사용하여 train/test값 나누기
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(100))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
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
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("loss : ", loss)
print('mae : ', mae)
print('RMSE : ', RMSE(y_test, y_predict))
print("R2 : ", r2)


# shuffle =True
# loss :  2.27009883113638e-10
# mae :  1.068115216185106e-05
# RMSE :  1.5542258291216555e-05
# R2 :  0.9999999999997221

# shuffle =False
# loss :  9.08039532454552e-10
# mae :  2.136230432370212e-05
# RMSE :  2.5984781442148437e-05
# R2 :  0.999999999979693

# shuffle 한게 결과값이 더 좋다
# shuffle 안하면 정해진 값으로만 훈련해서 범위 벗어나는 값 예측 힘들다











