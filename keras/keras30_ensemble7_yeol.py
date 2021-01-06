# 열이 다른 앙상블 모델에 대해 공부

import numpy as np
from numpy import array
# 1. 데이터
x1 = array([[1,2],[2,3],[3,4],[4,5],[5,6],
             [6,7],[7,8],[8,9],[9,10],[10,11],
             [20,30],[30,40],[40,50]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y1 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1_pred = array([55,65])
x2_pred = array([65,75,85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 모델 1
input1 = Input(shape=(2,1))
dense1 = LSTM(16, activation='relu')(input1)
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(80, activation='relu')(dense1)
dense1 = Dense(160, activation='relu')(dense1)
dense1 = Dense(80, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델 2
input2 = Input(shape=(3,1))
dense2 = LSTM(16, activation='relu')(input2)
dense2 = Dense(10)(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(32, activation='relu')(merge1)
middle1 = Dense(32)(middle1)


# 분기1
output1 = Dense(20)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

# 분기2
output2 = Dense(20)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(1)(output2)

 # 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1,output2]) # 2개 이상은 리스트로 묶는다
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit([x1, x2], [y1, y2], epochs=200, batch_size=16, callbacks=[early_stopping])

# 4. 평가, 예측
loss = model.evaluate([x1, x2], [y1, y2], batch_size=16)
x1_pred = x1_pred.reshape(1,2,1)
x2_pred = x2_pred.reshape(1,3,1)

y1_pred, y2_pred = model.predict([x1_pred, x2_pred])


print(y1_pred)
print(y2_pred)

# ValueError: Data cardinality is ambiguous:
#   x sizes: 10, 13
#   y sizes: 10, 13
# Please provide data which shares the same first dimension.
