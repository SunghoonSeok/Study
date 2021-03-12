# 인공지능계의 hello world라 불리는 mnist!!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow as tf
# # GPU 여러개일때 여러개 쓰는 방법
# gpus[0] 괄호 안에 숫자를 바꿔 gpu 선택, visual code 여러개 켜서 하나는 [0] 하나는 [1]로 따로 사용 가능
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus :
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     except RuntimeError as e:
#         print(e)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  # 전처리
x_test = x_test.reshape(10000, 28, 28, 1)/255.  # 전처리

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto', factor=0.5)
import tensorflow as tf
from tensorflow.keras.regularizers import l1, l2, l1_l2
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same',strides=1, input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Conv2D(64, 2, kernel_initializer='he_normal'))
# relu 계열에서는 he normal, sigmoid tanh계열은 xavier

model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32, 2, kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=128, epochs=7, validation_split=0.2, callbacks=[early_stopping, cp, lr])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# print("y_test : ", y_test[:-10])
# print("y_pred : ", y_recovery)
print("loss : ", result[0])
print("accuracy : ", result[1])
