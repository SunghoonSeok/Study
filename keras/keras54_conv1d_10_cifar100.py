import numpy as np

x_train = np.load('../data/npy/cifar100_x_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')


x_train = x_train.reshape(x_train.shape[0], 32*32, 3).astype('float32')/255.  # 전처리
x_test = x_test.reshape(x_test.shape[0], 32*32, 3).astype('float32')/255.  # 전처리
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                 strides=1,activation='relu', input_shape=(32*32, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
modelpath= '../data/modelcheckpoint/k54_conv1d_cifar100_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=64)
y_pred = model.predict(x_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)

# loss :  2.576901435852051
# acc :  0.36959999799728394