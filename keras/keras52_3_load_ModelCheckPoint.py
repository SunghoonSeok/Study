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

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model2 = Sequential()
# model2.add(Conv2D(filters=100, kernel_size=(4, 4), padding='same',
#                  strides=1, input_shape=(28, 28, 1)))
# model2.add(MaxPooling2D(pool_size=6))
# # model2.add(Dropout(0.2))
# model2.add(Conv2D(50, 2))
# model2.add(Flatten())
# model2.add(Dense(40, activation='relu'))
# model2.add(Dense(30, activation='relu'))
# model2.add(Dense(10, activation='relu'))
# model2.add(Dense(10, activation='softmax'))

# model.save('../data/h5/k52_1_model1.h5')

# 3. 컴파일, 훈련
# model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
# modelpath= '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# hist = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2, callbacks=[early_stopping])#, cp])

# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')


# 4-1. 평가, 예측
model = load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')
result = model.evaluate(x_test, y_test, batch_size=128)

print("로드체크포인트_loss : ", result[0])
print("로드체크포인트_accuracy : ", result[1])



# # 4-2. 평가, 예측
# model2.load_weights('../data/h5/k52_1_weight.h5')
# result2 = model2.evaluate(x_test, y_test, batch_size=128)

# print("가중치_loss : ", result2[0])
# print("가중치_accuracy : ", result2[1])


# 로드모델_loss :  0.048493627458810806
# 로드모델_accuracy :  0.98580002784729

# 가중치_loss :  0.048493627458810806
# 가중치_accuracy :  0.98580002784729   -> early stopping patience 적용됨

# 로드체크포인트_loss :  0.04041830822825432
# 로드체크포인트_accuracy :  0.9872999787330627 -> loss 최저점