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


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 4, 1)
x_test = x_test.reshape(x_test.shape[0], 4, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(200, 1, padding='same', input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
modelpath= '../data/modelcheckpoint/k54_conv1d_iris_checkpoint.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# 0.0001104229231714271 1.0


# Dropout 적용후
# 0.2407277524471283 0.9333333373069763
# 0.10077794641256332 0.9666666388511658

# loss :  0.06372033059597015
# acc :  1.0