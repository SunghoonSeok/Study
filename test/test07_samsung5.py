import numpy as np
import pandas as pd

data = np.load('./test/samsung_data.npy')
x = data[:,:-1]
y = data[:,-1]

print(x.shape, y.shape) # (2397, 14) (2397,)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)



size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x_data = split_x(x, size)
print(x.shape)

x = x_data[:-1,:,:]
y = y[6:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

x_pred = x_data[-1,:,:]
x_pred = x_pred.reshape(1,6,14)
print(x_train.shape, x_test.shape) # (1912, 6, 14) (479, 6, 14)
print(y_train.shape, y_test.shape) # (1912, )  (479,)
print(x_pred.shape) # (1, 6, 14)




#2. 모델구성
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(6,14))
dense1 = LSTM(512)(inputs)
dense1 = Dense(256)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(16)(dense1)
dense1 = Dense(8)(dense1)
dense1 = Dense(4)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping(monitor='val_loss', patience=60, mode='auto')
modelpath= './test/samsung5_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=64, epochs=500, validation_split=0.2, callbacks=[cp])

model.save('./test/samsung5_model.h5')

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_predict = model.predict(x_pred)
print(y_predict)

# loss, mae :  1428569.375 895.7993774414062
# RMSE :  1195.2277438453937
# R2 :  0.9747082894609304
# [[89665.516]]

# loss, mae :  1283006.25 889.0049438476562
# RMSE :  1132.6986843189306
# R2 :  0.9772853701647488
# [[92528.85]]

# loss, mae :  1323854.625 901.7225952148438
# RMSE :  1150.5888599829568
# R2 :  0.9765621807110992
# [[92427.43]]