import numpy as np


# 1. 데이터
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()


x = datasets.data
y = datasets.target
print(x.shape, y.shape)  #(569, 30)  (569,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 6, 5, 1)
x_test = x_test.reshape(x_test.shape[0], 6, 5, 1)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(200, 2, padding='same', input_shape=(6, 5, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(150, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid')) # 이진분류일때 마지막 activation은 반드시 sigmoid

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # 이진분류일때 loss는 binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print("loss : ", loss)
print("acc : ", acc)


# 결과치 나오게 코딩할것 0또는 1로

# loss= 0.046165917068719864 
# acc =0.9912280440330505

# Dropout 적용후
# 0.06107158586382866 0.9912280440330505
# 0.056601207703351974 0.9824561476707458

# LSTM
# 0.09797193855047226 0.9824561476707458

# CNN
# loss :  0.15443535149097443
# acc :  0.9649122953414917

# kernel size 1, patience 20
# loss :  0.07676749676465988
# acc :  0.9736841917037964

# kernel size 4, maxpooling 삭제, dropout 적용
# loss :  0.04616907238960266
# acc :  0.9824561476707458