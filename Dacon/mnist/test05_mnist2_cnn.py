import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, Input

from pandas import read_csv
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
submission = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)
print(submission.shape) # (20480, 2)

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)
x = temp.iloc[:,3:]/255
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y1 = y.to_numpy()
x_pred = x_test.to_numpy()



x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size=0.1, shuffle=True)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_pred = x_pred.reshape(-1, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

# 2. 모델 구성
# model = Sequential()
# model.add(Conv2D(256, 2, padding='same', activation='relu', input_shape=(28,28,1)))
# model.add(Conv2D(128,2))
# model.add(AveragePooling2D(pool_size=2))
# model.add(Conv2D(64, 2))
# model.add(Conv2D(32, 2))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))




# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 15)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
filepath = f'c:/data/test/mnist/checkpoint/mnist_checkpoint1.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
optimizer = Adam(lr=0.002)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), batch_size=32, callbacks=[es,lr,cp])


# 4. 평가, 예측
model2 = load_model('c:/data/test/mnist/checkpoint/mnist_checkpoint1.hdf5')
y_pred = model2.predict(x_pred)
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)


submission.iloc[:,1] = y_recovery
submission.to_csv('c:/data/test/mnist/submission_mnist2.csv', index=False)  