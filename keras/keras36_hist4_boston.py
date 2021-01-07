import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape) # (506, 13)
# print(y.shape) # (506,)
# print("==================")
# print(x[:5])
# print(y[:10])

# print(np.max(x), np.min(x)) # 711.0  0.0
# print(dataset.feature_names)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
inputs = Input(shape=(13,))
dense1 = Dense(64, activation='relu')(inputs)
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

hist = model.fit(x_train, y_train, batch_size=16, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
plt.show()


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=16)
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

# loss, mae :  10.149385452270508 2.244107484817505
# RMSE :  3.185810061249081
# R2 :  0.8785710807689575