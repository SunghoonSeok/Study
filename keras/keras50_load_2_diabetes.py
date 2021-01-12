import numpy as np

x = np.load('../data/diabetes_x.npy')
y = np.load('../data/diabetes_y.npy')


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
from tensorflow.keras.layers import Dense, Input
inputs = Input(shape=(10,))
dense1 = Dense(8, activation='linear')(inputs)
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
# modelpath= './modelCheckpoint/k46_5_diabetes_{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
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

# loss, mae :  2578.749267578125 39.9632682800293
# RMSE :  50.78138772576483
# R2 :  0.5208084824108226