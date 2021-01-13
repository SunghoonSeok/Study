import numpy as np

x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')


x_train = x_train.reshape(60000, 28*28, 1).astype('float32')/255.  # 전처리
x_test = x_test.reshape(10000, 28*28, 1)/255.  # 전처리

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=4, padding='same',
                 strides=1, input_shape=(28*28, 1)))
model.add(MaxPooling1D(pool_size=6))
# model.add(Dropout(0.2))
model.add(Conv1D(50, 2))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= '../data/modelcheckpoint/k54_conv1d_mnist_checkpoint.hdf5'
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

# loss :  0.040886472910642624
# accuracy :  0.9866999983787537

# poolsize=6
# loss :  0.05597861856222153
# acc :  0.9905999898910522

