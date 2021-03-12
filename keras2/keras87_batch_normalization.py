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

# initializer & regularizer

# kernel_ initializer(weight initializer) 
# : 가중치를 잘못 설정하면 기울기소실, 표현력 한계등 여러 문제를 야기하기에 초기값 설정은 중요하다. 

# - Xaiver : initializer에 여러 방법이 있지만 대부분 최근나온 Xavier, He를 많이 쓴다.
#            이전 노드와 다음 노드의 개수에 의존하는 방식. 비선형함수(sigmoid, tanh)에는 효과적이나
#            relu사용시 출력값이 0으로 수렴하게 되는 현상이 발생한다.
# - He : Xavier의 이러한 문제점을 해결하기 위한 초기화 방법이다. Xavier에서 출력 노드를 제거하고 사용한다.

# bias_initializer
# : bias 초기값 또한 중요하다. 보통 0으로 초기화하지만 relu의 경우 0.01과 같은 작은 값으로 초기화 한다고 한다.

# kernel_regularizer (weight regularizer)
# : 네트워크의 복잡도에 제한을 두어 가중치가 작은 값을 가지도록 강제하는 것(발산, 과적합을 막기 위함)
# - L1 : 가중치의 절댓값에 비례하는 비용이 추가됨
# - L2(0.001) : 가중치의 제곱에 비례하는 비용이 추가됨, 제곱후 괄호안의 숫자를 곱하는 방식
# L1과 L2의 가장 큰 차이점은 L1은 대상 크기에 관계없이 일정한 힘으로 밀기에 0이 되기도 하지만
# L2는 대상이 0에 가까워 질수록 미는 힘도 작아져 아주 작은 값이 될 지언정 0이 되진 않는다.

# Dropout : 입력 벡터 중 일부를 무작위로 0으로 바꾼다. 테스트 단계에선 어떤 유닛도 드롭아웃 되지 않으나
#           대신 층의 출력을 드롭아웃 비율에 비례해서 줄여준다.

# BatchNormalization : mini-batch의 평균과 분산으로 normalize하고 test할때는 계산해놓은 이동 평균으로 normalize한다.
#                      장점은 parameter scale에 영향을 받지 않게 되어 learning rate를 크게 잡을 수 있게 되고
#                      상대적으로 느린 Dropout을 제외할 수 있게 된다.(Dropout의 효과가 BN의 효과와 같이에)