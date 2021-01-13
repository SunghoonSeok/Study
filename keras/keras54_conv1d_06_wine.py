import numpy as np

x = np.load('../data/npy/wine_x.npy')
y = np.load('../data/npy/wine_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(200, 2, padding='same', input_shape=(13, 1)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) 

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
modelpath= '../data/modelcheckpoint/k54_conv1d_wine_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.10290367901325226
# acc :  0.9722222089767456


# conv1d
# loss :  0.005717820022255182
# acc :  1.0