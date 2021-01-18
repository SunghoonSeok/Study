import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Nadam(lr=0.1)
# loss : 1.0687887197491364e-06 결과물 : [[10.9994545]]
# optimizer = Nadam(lr=0.01)
# loss : 2.1316282072803006e-13 결과물 : [[10.999998]]
# optimizer = Nadam(lr=0.001)
# loss : 0.0001348371442873031 결과물 : [[11.008814]]
# optimizer = Nadam(lr=0.0001)
# loss : 1.1585811989789363e-06 결과물 : [[10.998624]]



# optimizer = SGD(lr=0.1)
# loss : nan 결과물 : [[nan]]
# optimizer = SGD(lr=0.01)
# loss : nan 결과물 : [[nan]]
# optimizer = SGD(lr=0.001)
# loss : 4.9664055978837496e-08 결과물 : [[10.999603]]
# optimizer = SGD(lr=0.0001)
# loss : 0.0012544934870675206 결과물 : [[10.956595]]


# optimizer = RMSprop(lr=0.1)
# loss : 950.1724853515625 결과물 : [[77.37674]]
# optimizer = RMSprop(lr=0.01)
# loss : 108.4044189453125 결과물 : [[27.448935]]
# optimizer = RMSprop(lr=0.001)
# loss : 0.13246223330497742 결과물 : [[11.469002]]
# optimizer = RMSprop(lr=0.0001)
# loss : 0.0002950725320260972 결과물 : [[10.964246]]



# optimizer = Adagrad(lr=0.1)
# loss : 15.202878952026367 결과물 : [[5.882399]]
# optimizer = Adagrad(lr=0.01)
# loss : 1.6611602404736914e-07 결과물 : [[10.999759]]
# optimizer = Adagrad(lr=0.001)
# loss : 1.1661339158308692e-05 결과물 : [[10.993397]]
# optimizer = Adagrad(lr=0.0001)
# loss : 0.004074796102941036 결과물 : [[10.920638]]


# optimizer = Adamax(lr=0.1)
# loss : 1.3207681615057254e-09 결과물 : [[11.000063]]
# optimizer = Adamax(lr=0.01)
# loss : 2.7853275783897014e-13 결과물 : [[10.999999]]
# optimizer = Adamax(lr=0.001)
# loss : 1.0909282082138816e-06 결과물 : [[10.998946]]
# optimizer = Adamax(lr=0.0001)
# loss : 0.0021568224765360355 결과물 : [[10.940715]]



# optimizer = Adam(lr=0.1)
# loss : 5.7526769126070576e-08 결과물 : [[11.000243]]
# optimizer = Adam(lr=0.01)
# loss : 2.3483436874770225e-13 결과물 : [[11.000001]]
optimizer = Adam(lr=0.001)
# loss : 1.8474111129762605e-13 결과물 : [[10.999999]]
# optimizer = Adam(lr=0.0001)
# loss : 8.256919500126969e-06 결과물 : [[10.994189]]

# optimizer = Adadelta(lr=0.1)
# loss : 3.9255382944247685e-06 결과물 : [[10.995953]]
# optimizer = Adadelta(lr=0.01)
# loss : 0.00012576306471601129 결과물 : [[11.016877]]
# optimizer = Adadelta(lr=0.001)
# loss : 7.624326229095459 결과물 : [[6.042479]]
# optimizer = Adadelta(lr=0.0001)
# loss : 23.256515502929688 결과물 : [[2.443701]]


model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print("loss :", loss, "결과물 :", y_pred)