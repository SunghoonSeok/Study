# keras21_cancer1.py 를 다중분류로 코딩하시오.
import numpy as np


# 1. 데이터
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

print(datasets.feature_names)
print(datasets.DESCR)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30)  (569,)
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


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(30,)))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val), callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

# 실습 1. acc 0.985이상 올릴것
# 실습 2. predict 출력해 볼것


y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)


