# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성하시오.

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x[:5])
# print(y[:10])
# print(x.shape, y.shape)  # (442, 10)  (442,)
# print(np.max(x), np.min(x))
# print(dataset.feature_names)
# print(dataset.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.6, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 10, 1)
x_val = x_val.reshape(-1, 10, 1)
x_test = x_test.reshape(-1, 10, 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(10,1))
dense1 = LSTM(20, activation='relu')(inputs)
dense1 = Dense(16)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(30)(dense1)
dense1 = Dense(20)(dense1)
dense1 = Dense(10)(dense1)
dense1 = Dense(5)(dense1)
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



# 1
# loss, mae :  6990.39404296875 63.32801055908203
# RMSE :  83.60857604121809
# R2 :  -0.0770956046409785

# 2
# loss, mae :  6896.97119140625 66.08655548095703
# RMSE :  83.04800659925694
# R2 :  -0.0627008425459401


# 3
# loss, mae :  6688.025390625 68.48439025878906
# RMSE :  81.78034462614477
# R2 :  -0.030505876090359818

# 4
# loss, mae :  5792.2197265625 60.13637161254883
# RMSE :  76.10663409909971
# R2 :  0.10752177215403858

# 5
# loss, mae :  5414.12890625 57.77206039428711
# RMSE :  73.58076590906916
# R2 :  0.16577882740916317

# 6
# loss, mae :  5032.68994140625 58.31814956665039
# RMSE :  70.94145393061724
# R2 :  0.2245518399835802

# LSTM
# loss, mae :  4179.7392578125 55.49616622924805
# RMSE :  64.65090026540445
# R2 :  0.2848301038186290

# loss, mae :  3840.151611328125 50.478023529052734
# RMSE :  61.96895904702104
# R2 :  0.39017286950091945


# LSTM
# loss, mae :  3977.69775390625 52.91457748413086
# RMSE :  63.06899168420695
# R2 :  0.4023517803765174

# loss, mae :  3844.3203125 50.2368278503418 -> relu
# RMSE :  62.00258535152787
# R2 :  0.40765844987626576