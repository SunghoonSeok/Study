# 실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지
# 총 6개의 파일을 완성하시오.

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x= scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
inputs = Input(shape=(10,))
dense1 = Dense(4, activation='relu')(inputs)
dense1 = Dense(8, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(16, activation='relu')(dense1)
dense1 = Dense(18, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)
dense1 = Dense(4, activation='relu')(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=400, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=4)
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




