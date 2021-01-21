import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

def quantile_loss(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
def quantile_loss_dacon(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 1. 데이터
x = np.array([1.,2.,3.,4.,5.,6.,7.,8.])
y = np.array([1.,2.,3.,4.,5.,6.,7.,8.])
x_pred = np.array([9.])
# 2. 모델구성
a=[]
for q in quantiles:
    model = Sequential()
    model.add(Dense(100, input_shape=(1,)))
    model.add(Dense(50))
    model.add(Dense(30))
    model.add(Dense(1))

    # model.compile(loss=quantile_loss, optimizer='adam')
    model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(q, y_true, y_pred), optimizer='adam')
    model.fit(x,y,batch_size=1,epochs=30,verbose=0)
    loss = model.evaluate(x,y)
    y_pred = model.predict(x_pred)
    a.append(y_pred)

print(a)





print(loss)