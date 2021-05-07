# a = 1
# b = 2
# c = a + b
# print(c)

# import tensorflow as tensorflow
# import keras
# import numpy as np
# # print("잘 설치됐다.")
# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])

# x = x[:,:2]
# print(x)
# import numpy as np
# import pandas as pd
# x = np.array([[0.254534, 0.4235656, 13.4423, 15.35363],[13.5345634,41.5234534,13.53456,55.45345]])
# x = x.round(2)
# # x = np.round(x, 2)
# print(x)
# x = np.array(range(9))
# print(x)

# for i in range(9):
#     print('%d'%(i+1))

# x = np.array([1,2,3,4,5,6,7,8,9])
# print(x.shape)
# df = pd.DataFrame({'cat': ['A','A','A','A','A','B','B','B','B','B','B'],
#                    'sales': [10, 20, 30, 40, 50, 1, 2, 3, 4, 5,6]})
# df['sales'].quantile(q=0.5, interpolation='nearest')

# print(df['sales'].quantile(q=0.5, interpolation='nearest')) # 5
# df.groupby(['cat'])['sales'].quantile(q=0.50, interpolation='nearest')

# print(df.groupby(['cat'])['sales'].quantile(q=0.50, interpolation='nearest'))
# cat
# A    30
# B     3
# Name: sales, dtype: int64

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(range(100))
y = np.array(range(100))
x_pred = np.array(range(100,200))

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8)

model = Sequential()
model.add(Dense(32,input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from tensorflow.keras.callbacks import EarlyStopping
a=[]
for q in quantiles
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer = optimizer, metrics = [lambda y,y_pred: quantile_loss(q,y,y_pred)])
    es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    hist = model.fit(x,y, validation_data=(x_val,y_val), epochs=100, batch_size=4, callbacks=[es])
    a.append(hist.history['val_loss'][-11])
print(a)
    