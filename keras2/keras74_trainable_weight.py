import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

# print(model.weights)
# print(model.trainable_weights) # 여기선 model.weights 랑 같다

print(len(model.weights)) # weight,bias가 각각 4개라 8개
print(len(model.trainable_weights))

