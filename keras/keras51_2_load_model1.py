import numpy as np

x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')


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

# model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(4, 4), padding='same',
#                  strides=1, input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=6))
# # model.add(Dropout(0.2))
# model.add(Conv2D(50, 2))
# model.add(Flatten())
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.summary()
model = load_model('../data/h5/k51_1_model1.h5')

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, batch_size=128, epochs=7, validation_split=0.2, callbacks=[early_stopping, cp])

# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# print("y_test : ", y_test[:-10])
# print("y_pred : ", y_recovery)
print("loss : ", result[0])
print("accuracy : ", result[1])


'''
# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)   # 2행 1열중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()


plt.title('한글되나')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) #2행 1열중 두번째
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()

plt.title('되야한다')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

'''

# loss :  0.040886472910642624
# accuracy :  0.9866999983787537