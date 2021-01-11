# 2차원을 4차원으로 늘여서 하시오.

import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D,Flatten, Dropout
inputs = Input(shape=(13, 1, 1))
dense1 = Conv2D(200, (3,1), padding='same')(inputs)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, batch_size=8, epochs=1000, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
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


# parameter조정값 -> layer와 node를 줄이고 batch_size와 epochs도 함께 줄임
# loss, mae :  6.998237133026123 1.869640827178955
# RMSE :  2.645418213195852
# R2 :  0.9162719360420687

# early stopping 적용후
# loss, mae :  6.493799686431885 2.0575599670410156
# RMSE :  2.5482935370353137
# R2 :  0.9223071100610187

# dropout 적용
# loss, mae :  6.390140533447266 1.969919204711914
# RMSE :  2.527872717540827
# R2 :  0.9235473090551892

# LSTM
# loss, mae :  9.202030181884766 2.191269874572754
# RMSE :  3.033484837548874
# R2 :  0.8899053975201897

# CNN
# loss, mae :  22.5207576751709 3.0655016899108887
# RMSE :  4.7456038999558094
# R2 :  0.730557969195601

# loss, mae :  16.587936401367188 2.9508514404296875
# RMSE :  4.072828808230732
# R2 :  0.8015392252212221

# Conv2D 200, kernel size (3,1) Dropout 적용
# loss, mae :  15.561871528625488 2.997842788696289
# RMSE :  3.944854039521248
# R2 :  0.813815188961931