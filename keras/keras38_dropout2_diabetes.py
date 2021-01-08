
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape)  # (442, 10)  (442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.6, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)



#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
inputs = Input(shape=(10,))
dense1 = Dense(8, activation='linear')(inputs)
dense1 = Dense(16)(dense1)
dense1 = Dense(32)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
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
model.fit(x_train, y_train, batch_size=8, epochs=500, validation_data=(x_val, y_val), callbacks=[early_stopping])

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

# 5
# loss, mae :  5414.12890625 57.77206039428711
# RMSE :  73.58076590906916
# R2 :  0.16577882740916317

# 6
# loss, mae :  5032.68994140625 58.31814956665039
# RMSE :  70.94145393061724
# R2 :  0.2245518399835802

# dropout 적용
# loss, mae :  2467.35888671875 39.37063980102539
# RMSE :  49.672514799353344
# R2 :  0.5710710900945571

# loss, mae :  3425.866455078125 48.849830627441406
# RMSE :  58.53090289934952
# R2 :  0.4829418691025006