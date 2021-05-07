import numpy as np
import pandas as pd

# 1. 데이터
data1 = np.load('c:/data/test/samsung_jusik_all.npy')
data2 = np.load('c:/data/test/kodex_jusik_all.npy')

x1 = data1[1314:,:-1]
y = data1[1314:,-1]
x2 = data2
print(x1.shape, y.shape, x2.shape)


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(x1)
x1 = scaler1.transform(x1)

scaler2 = MinMaxScaler()
scaler2.fit(x2)
x2 = scaler2.transform(x2)

size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x1_data = split_x(x1, size)
x2_data = split_x(x2, size)

x1 = x1_data[:-2,:,:]
x2 = x2_data[:-2,:,:]
y = y[7:]
print(x1.shape, y.shape, x2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=66)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
# inputs = Input(shape=(6, x.shape[2]))
# dense1 = Conv1D(1000, 2, padding='same', activation='relu')(inputs)
# dense1 = MaxPooling1D(pool_size=2)(dense1)
# dense1 = Conv1D(500, 2, activation='relu')(dense1)
# dense1 = Conv1D(400, 2,activation='relu')(dense1)
# dense1 = Flatten()(dense1)

# 모델1
input1 = Input(shape=(6,x1.shape[2]))
dense1 = LSTM(1000)(input1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Dense(500)(dense1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Dense(400)(dense1)
dense1 = Dense(300)(dense1)
dense1 = Dense(200)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(10)(dense1)

# 모델2
input2 = Input(shape=(6,x1.shape[2]))
dense2 = GRU(1000)(input2)
# dense2 = Dropout(0.2)(dense2)
dense2 = Dense(500)(dense2)
# dense2 = Dropout(0.2)(dense2)
dense2 = Dense(400)(dense2)
dense2 = Dense(300)(dense2)
dense2 = Dense(200)(dense2)
dense2 = Dense(100)(dense2)
dense2 = Dense(50)(dense2)
dense2 = Dense(30)(dense2)
dense2 = Dense(10)(dense2)


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(128)(merge1)
middle1 = Dense(64)(middle1)
middle1 = Dense(32)(middle1)
middle1 = Dense(16)(middle1)
middle1 = Dense(8)(middle1)
middle1 = Dense(4)(middle1)
middle1 = Dense(2)(middle1)
outputs = Dense(1)(middle1)





model = Model(inputs=[input1,input2], outputs=outputs)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=80, mode='auto')
modelpath= 'c:/data/test/samsung6_checkpoint2_{val_loss}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
reduce_lr =ReduceLROnPlateau(monitor='val_loss', patience=40, factor=0.5, verbose=1)
model.fit([x1_train,x2_train], y_train, batch_size=64, epochs=1000, validation_split=0.2, callbacks=[cp, es, reduce_lr])

model.save('c:/data/test/samsung6_model2.h5')

#4. 평가, 예측
loss, mae = model.evaluate([x1_test,x2_test], y_test, batch_size=64)
y_predict = model.predict([x1_test,x2_test])
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

x1_pred = x1_data[-2:,:,:]
x2_pred = x2_data[-2:,:,:]
x1_pred = x1_pred.reshape(x1_pred.shape[0],6,x1_pred.shape[-1])
x2_pred = x2_pred.reshape(x2_pred.shape[0],6,x2_pred.shape[-1])

y_predict = model.predict([x1_pred,x2_pred])

print(y_predict)


print("월요일 삼성 시가 : ", int(np.round(y_predict[0])), "원")
print("화요일 삼성 시가 : ", int(np.round(y_predict[1])), "원")


# 로드 모델
#loss, mae :  2541032.0 1262.46630859375
# RMSE :  1594.061363025406
# R2 :  0.9729504677582311
# [[91914.98]
#  [88692.26]]
# 월요일 삼성 시가 :  91915 원
# 화요일 삼성 시가 :  88692 원

# check
# loss, mae :  2288014.0 1166.26708984375
# RMSE :  1512.6178212770928
# R2 :  0.9756438794913104
# [[90409.25]
#  [88900.08]]
# 월요일 삼성 시가 :  90409 원
# 화요일 삼성 시가 :  88900 원