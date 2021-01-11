# (N, 28, 28) -> (N, 764) -> (N, 764, 1)
# 주말 과제
# lstm 모델로 구성  input_shape=(28*28, 1) ->(28*14, 2) (7*7, 16)

# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) # (28, 28)

x_train = x_train.astype('float32')/255.  # 전처리
x_test = x_test.astype('float32')/255.  # 전처리

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

model = Sequential()
model.add(LSTM(50, input_shape=(28, 28)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, batch_size=128, epochs=70, validation_split=0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)


# poolsize=7, conv2d 두번째 50, 첫 dense=40
# loss :  0.06115453317761421
# acc :  0.9900000095367432

# poolsize=6
# loss :  0.05597861856222153
# acc :  0.9905999898910522

# DNN
# loss :  0.2799952030181885
# acc :  0.9646999835968018

# loss :  0.25460025668144226
# acc :  0.9643999934196472

# DNN node값 수정 - 80,60,40,20,10,10
# loss :  0.17857107520103455
# acc :  0.9696000218391418

# loss :  0.1557396650314331
# acc :  0.9771999716758728

# LSTM
# loss :  0.11201491951942444
# acc :  0.9778000116348267

# LSTM activation 수정
# loss :  0.09609527885913849
# acc :  0.9811999797821045

# lstm node 100
# loss :  0.06158873811364174
# acc :  0.989300012588501