import numpy as np
from numpy import array
# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],
             [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1_predict = array([55, 65, 75])
x2_predict = array([65,75,85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 모델 1
input1 = Input(shape=(3,1))
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
dense2 = Dense(1)(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
output1 = Dense(32)(merge1)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)



 # 모델 선언
model = Model(inputs=[input1, input2], outputs=output1) # 2개 이상은 리스트로 묶는다
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit([x1, x2], y, epochs=2000, batch_size=16, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate([x1, x2], y, batch_size=16)


x1_pred = x1_predict.reshape(1,3,1)
x2_pred = x2_predict.reshape(1,3,1)



y_pred = model.predict([x1_pred, x2_pred])


print(y_pred)

# loss: 1.3719e-04
# [[79.62926]]

# loss: 0.0322
# [[79.276566]]

# loss: 7.2875e-06
# [[81.55755]]

# loss: 4.2627e-04 -> x2와 merge layer 줄이기
# [[86.062164]]

# loss: 4.5744e-04
# [[86.3096]]

# loss: 1.9322e-04
# [[84.91208]]

# loss: 4.3607e-04
# [[85.73882]]

# loss: 4.1401e-05
# [[84.49505]]

# loss: 5.0610e-04
# [[84.82469]]