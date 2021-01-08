import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(np.max(x), np.min(x)) # 711.0  0.0
print(dataset.feature_names)

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
from tensorflow.keras.layers import Dense, Input, Dropout
inputs = Input(shape=(13,))
dense1 = Dense(64, activation='relu')(inputs)
dense1 = Dense(128)(dense1)
dense1 = Dropout(0.2)(dense1)
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
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, batch_size=8, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

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





