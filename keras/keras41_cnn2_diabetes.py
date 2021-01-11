
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target


print(x.shape, y.shape)  # (442, 10)  (442,)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],5, 2, 1)
x_test = x_test.reshape(x_test.shape[0],5, 2, 1)

print(x_train.shape)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
inputs = Input(shape=(5, 2, 1))
dense1 = Conv2D(100, 2, padding='same')(inputs)
dense1 = MaxPooling2D(pool_size=2)(dense1)
dense1 = Flatten()(dense1)
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
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, batch_size=8, epochs=500, validation_split=0.2, callbacks=[early_stopping])

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


# loss, mae :  5032.68994140625 58.31814956665039
# RMSE :  70.94145393061724
# R2 :  0.2245518399835802

# loss, mae :  3199.23486328125 46.428470611572266 -> graph를 보고 과적합 부분을 제거한 결과값
# RMSE :  56.56178118910292
# R2 :  0.4737240603633286

# loss, mae :  2578.378173828125 40.707279205322266
# RMSE :  50.777731332389855
# R2 :  0.5364073936410592

# LSTM
# loss, mae :  3977.69775390625 52.91457748413086
# RMSE :  63.06899168420695
# R2 :  0.4023517803765174

# loss, mae :  3844.3203125 50.2368278503418 -> relu
# RMSE :  62.00258535152787
# R2 :  0.40765844987626576

# CNN
# loss, mae :  4197.63525390625 53.850521087646484
# RMSE :  64.78916241484038
# R2 :  0.3532188848774409

# maxpooling 삭제, patience 20, kernel size 4
# loss, mae :  3513.9150390625 47.440616607666016
# RMSE :  59.27828603643157
# R2 :  0.45856805514012333