import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터

# x, y = load iris(return_X_y=True) -> 교육용 데이터에선 가능

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape, y.shape) #(150, 4),  (150, )
# print(x[:5])
# print(y)
x_pred = x[-5:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 4, 1)
x_val = x_val.reshape(-1, 4, 1)
x_test = x_test.reshape(-1, 4, 1)
x_pred = x_pred.reshape(-1, 4, 1)


# print(y)
# print(x.shape) # (150,4)
# print(y.shape) # (150,3)



# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val), batch_size=16, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print(loss, acc)

y_pred = model.predict(x_pred)
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
# 결과치 나오게 코딩할것 # argmax



# 0.1297154575586319 0.9666666388511658
# [[3.3373621e-10 3.2250402e-03 9.9677497e-01]
#  [5.2666786e-07 1.2779574e-01 8.7220377e-01]
#  [1.2780142e-08 2.5820270e-02 9.7417974e-01]
#  [3.1074665e-10 3.5898173e-03 9.9641019e-01]]

# 0.011982857249677181 1.0
# [[2.1100654e-12 1.2142000e-03 9.9878579e-01]
#  [3.1866403e-09 3.8676243e-02 9.6132374e-01]
#  [9.8467144e-11 7.1519045e-03 9.9284804e-01]
#  [6.6156330e-12 2.1690922e-03 9.9783093e-01]]