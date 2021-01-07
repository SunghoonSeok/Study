import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터

# x, y = load iris(return_X_y=True) -> 교육용 데이터에선 가능

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape, y.shape) #(150, 4),  (150, )
print(x[:5])
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y)
print(x.shape) # (150,4)
print(y.shape) # (150,3)



# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(4,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
hist = model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val), callbacks=[early_stopping])

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
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
# 결과치 나오게 코딩할것 # argmax

# 0.002801266498863697 1.0
# [[0.0000000e+00 4.0494348e-25 1.0000000e+00]
#  [0.0000000e+00 2.2388051e-23 1.0000000e+00]
#  [0.0000000e+00 1.1332989e-24 1.0000000e+00]
#  [0.0000000e+00 1.1197788e-25 1.0000000e+00]]

# 0.03470752760767937 0.9666666388511658
# [[0.0000000e+00 1.1491760e-16 1.0000000e+00]
#  [0.0000000e+00 2.9781458e-15 1.0000000e+00]
#  [0.0000000e+00 4.4913393e-16 1.0000000e+00]
#  [0.0000000e+00 4.8619284e-17 1.0000000e+00]]

# 0.0018509665969759226 1.0
# [[6.2294304e-37 3.6528328e-20 1.0000000e+00]
#  [4.8576592e-34 1.7516075e-18 1.0000000e+00]
#  [8.5207671e-36 1.6631836e-19 1.0000000e+00]
#  [1.9210618e-37 8.1843447e-21 1.0000000e+00]]