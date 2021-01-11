import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)
print(x_train[0])
print(y_train[0]) 
print(x_train[0].shape) #(32,32,3)
print(y_train[0:50]) #0~9


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2], 3).astype('float32')/255.  # 전처리
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2], 3).astype('float32')/255.  # 전처리
y_train = y_train.reshape(y_train.shape[0],)
y_test = y_test.reshape(y_test.shape[0],)

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM

model = Sequential()
model.add(LSTM(50, input_shape=(32*32, 3)))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)

# loss :  2.289151430130005
# acc :  0.11829999834299088