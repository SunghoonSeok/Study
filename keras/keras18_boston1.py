import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)
print("==================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.85, shuffle=True)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
inputs = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(inputs)
dense1 = Dense(100)(dense1)
dense1 = Dense(200)(dense1)
dense1 = Dense(400)(dense1)
dense1 = Dense(800)(dense1)
dense1 = Dense(400)(dense1)
dense1 = Dense(200)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val))

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


# loss, mae :  19.518239974975586 3.345139503479004
# RMSE :  4.4179451574786555
# R2 :  0.7664805756106845



