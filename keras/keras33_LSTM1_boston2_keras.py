import numpy as np

from sklearn.datasets import load_boston

import numpy as np

from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 13, 1)
x_val = x_val.reshape(x_val.shape[0], 13, 1)
x_test = x_test.reshape(x_test.shape[0], 13, 1)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(13,1))
dense1 = LSTM(64)(inputs)
dense1 = Dense(128)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
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

# early stopping 전
# loss, mae :  11.031598091125488 2.355877637863159
# RMSE :  3.3213850693202898
# R2 :  0.8674785107264735

# early stopping 후
# loss, mae :  15.436517715454102 2.6361069679260254
# RMSE :  3.9289335323222736
# R2 :  0.8145626496625189

# LSTM
# loss, mae :  9.202030181884766 2.191269874572754
# RMSE :  3.033484837548874
# R2 :  0.8899053975201897