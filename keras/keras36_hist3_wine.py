import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR) 
# print(dataset.feature_names)

x = dataset.data
y = dataset.target
# print(x) 
# print(y) # 다중분류
# print(x.shape, y.shape) #(178, 13) (178, )
x_pred = x[-5:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(-1, 13, 1)
x_val = x_val.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
x_pred = x_pred.reshape(-1, 13, 1)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(200, input_shape=(13,1)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, epochs=512, validation_data=(x_val, y_val), callbacks=[early_stopping])

print(hist)
print(hist.history.keys()) # loss, acc, val_loss, val_acc


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


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print(loss, acc)

y_pred = model.predict(x_pred)
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)

# DNN
# 0.06456967443227768 0.9444444179534912
# [[3.0522069e-09 2.2839174e-10 1.0000000e+00]
#  [2.5904251e-08 5.4497056e-09 1.0000000e+00]
#  [6.7940507e-09 4.8495258e-10 1.0000000e+00]
#  [4.5260311e-08 6.9336692e-09 1.0000000e+00]]

# 0.0747983455657959 0.9722222089767456
# [[2.54100012e-08 3.35555850e-09 1.00000000e+00]
#  [7.85034899e-08 1.39990242e-08 9.99999881e-01]
#  [1.06828955e-08 1.03460784e-09 1.00000000e+00]
#  [2.18311733e-08 4.05825151e-09 1.00000000e+00]]

# 0.002636339981108904 1.0
# [[5.8743876e-10 1.8263988e-08 1.0000000e+00]
#  [1.7772624e-09 6.1571924e-08 9.9999988e-01]
#  [2.8748004e-10 9.9194519e-09 1.0000000e+00]
#  [1.2881725e-09 3.7203854e-08 1.0000000e+00]]

# LSTM
# 0.13784639537334442 0.9722222089767456
# [[1.1656784e-11 1.2106997e-11 1.0000000e+00]
#  [2.2648550e-11 1.7470293e-11 1.0000000e+00]
#  [2.0782051e-11 6.0548697e-12 1.0000000e+00]
#  [3.3030137e-08 2.3503985e-09 1.0000000e+00]]

# 0.19390122592449188 0.9722222089767456
# [[5.09957311e-17 7.35391765e-08 9.99999881e-01]
#  [5.77546137e-19 1.12829985e-08 1.00000000e+00]
#  [7.80318828e-20 4.01351219e-09 1.00000000e+00]
#  [1.25520284e-18 1.53409427e-08 1.00000000e+00]]

# 0.04197926074266434 0.9722222089767456
# [[1.8853380e-06 2.3874216e-04 9.9975938e-01]
#  [1.0815563e-06 3.8533259e-04 9.9961358e-01]
#  [1.1457514e-06 5.6021707e-04 9.9943858e-01]
#  [1.5472402e-05 7.4699055e-03 9.9251467e-01]]

# 1.3455135558615439e-05 1.0
# [[4.2863382e-15 5.5638240e-15 1.0000000e+00]
#  [2.1333199e-14 2.6451235e-14 1.0000000e+00]
#  [2.6653814e-14 2.3102707e-14 1.0000000e+00]
#  [2.3870224e-11 4.7479125e-11 1.0000000e+00]]

# 0.0031589895952492952 1.0
# [[9.5686026e-10 6.9872840e-05 9.9993014e-01]
#  [7.5123657e-11 1.1204101e-05 9.9998879e-01]
#  [2.4083107e-11 5.7127982e-06 9.9999428e-01]
#  [4.3766052e-10 3.6099460e-05 9.9996388e-01]]