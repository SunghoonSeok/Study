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
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
'''
from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y)
print(x.shape) # (150,4)
print(y.shape) # (150,3)
'''


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(200, activation='relu', input_shape=(4,)))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

model = LinearSVC()



# 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
# model.fit(x_train, y_train, epochs=400, validation_data=(x_val, y_val), callbacks=[early_stopping])
model.fit(x_train, y_train)
# 4. 평가, 예측
result = model.score(x_test, y_test)
print(result)

# print(loss, acc)

# y_pred = model.predict(x[-5:-1])
# print(y_pred)
# print(y[-5:-1])
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# # 결과치 나오게 코딩할것 # argmax

