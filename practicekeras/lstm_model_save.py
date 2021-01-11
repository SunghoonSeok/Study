import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input

# 모델 구성
input1 = Input(shape=(4,1))
dense1 = LSTM(50, name='add_lstm')(input1)
dense1 = Dense(40, name='add_dense1')(dense1)
dense1 = Dense(30, name='add_dense2')(dense1)
dense1 = Dense(20, name='add_dense3')(dense1)
dense1 = Dense(10, name='add_dense4')(dense1)
output1 = Dense(3, activation='softmax')(dense1)

model = Model(inputs=input1, outputs=output1)

model.save("./model/save_lstmmodel.h5")