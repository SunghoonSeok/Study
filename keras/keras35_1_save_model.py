import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 2. 모델
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

# 모델 저장

model.save("../data/h5/save_keras35.h5")
model.save("../data/h5/save_keras35_4.h5")
model.save("..\data\h5\save_keras35_5.h5")
model.save("..\\data\\h5\\save_keras35_6.h5")

