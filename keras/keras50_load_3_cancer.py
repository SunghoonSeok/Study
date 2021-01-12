import numpy as np

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=67)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 30, 1, 1)
x_test = x_test.reshape(x_test.shape[0], 30, 1, 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(400, (3,1), padding='same', input_shape=(30, 1, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(200, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
# modelpath= '../data/modelcheckpoint/k46_6_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping], batch_size=16)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)

print("loss : ", loss)
print("acc : ", acc)

# loss :  0.14868971705436707
# acc :  0.9473684430122375
