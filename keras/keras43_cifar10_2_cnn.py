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


x_train = x_train.astype('float32')/255.  # 전처리
x_test = x_test.astype('float32')/255.  # 전처리
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same',
                 strides=1, input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(64, 2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.8975118398666382
# acc :  0.6958000063896179

# kernel 2
# loss :  0.9071490168571472
# acc :  0.7020000219345093

# loss :  0.972180962562561
# acc :  0.7027999758720398