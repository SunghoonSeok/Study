import numpy as np

x_train = np.load('../data/npy/fashion_x_train.npy')
x_test = np.load('../data/npy/fashion_x_test.npy')
y_train = np.load('../data/npy/fashion_y_train.npy')
y_test = np.load('../data/npy/fashion_y_test.npy')

x_train = x_train.reshape(60000, 28*28, 1).astype('float32')/255.  # 전처리
x_test = x_test.reshape(10000, 28*28, 1)/255.  # 전처리

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
model.add(Conv1D(filters=200, kernel_size=4, padding='same',
                 strides=1, input_shape=(28*28, 1)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
modelpath= '../data/modelcheckpoint/k54_conv1d_fashion_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=128, epochs=7, validation_split=0.2, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(x_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)

# Dropout 적용, conv2d 두번째 node 100
# loss :  0.4641130864620209
# acc :  0.9088000059127808

# kernel size 3, conv2d 두번째 삭제
# loss :  0.29550397396087646
# acc :  0.9121999740600586

# kernel size 4
# loss :  0.3636307418346405
# acc :  0.9128000140190125

# loss :  0.2975737154483795
# acc :  0.9135000109672546

