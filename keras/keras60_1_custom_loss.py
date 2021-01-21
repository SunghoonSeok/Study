import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def custom_mse(y, y_pred):
    return tf.math.reduce_mean(tf.square(y - y_pred))


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(1,)))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

# 3. 컴파일, 훈련



model.compile(loss=custom_mse, optimizer='adam')
model.fit(x,y,batch_size=1,epochs=30)
loss = model.evaluate(x,y)
print(loss)