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

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(13,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print(loss, acc)

y_pred = model.predict(x_pred)
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)

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

# 0.024275613948702812 0.9722222089767456
# [[8.4655755e-10 5.1032600e-10 1.0000000e+00]
#  [5.1353584e-09 3.5814149e-09 1.0000000e+00]
#  [6.7729833e-10 4.1818604e-10 1.0000000e+00]
#  [2.6049707e-09 2.2014897e-09 1.0000000e+00]]