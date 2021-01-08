# (N, 28, 28) -> (N, 28*28) -> (28*28, )
# 주말 과제
# dense 모델로 구성 input_shape = (28*28, )

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

# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.  # 전처리
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.  # 전처리

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(784,)))
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


# poolsize=6
# loss :  0.05597861856222153
# acc :  0.9905999898910522

# DNN
# loss :  0.2799952030181885
# acc :  0.9646999835968018