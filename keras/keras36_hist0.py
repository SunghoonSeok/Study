import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:, :-1]
y = dataset[:, -1]

x = x.reshape(x.shape[0], x.shape[1], 1)

pred = split_x(range(96,106),5)
x_pred = pred[:, :-1]

# 2. 모델 구성
model = load_model("../data/h5/save_keras35.h5")
model.add(Dense(5,name='add_model_1'))
model.add(Dense(1,name='add_model_2'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x, y, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
print(hist)
print(hist.history.keys()) # loss, acc, val_loss, val_acc

print(hist.history['loss'])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()


