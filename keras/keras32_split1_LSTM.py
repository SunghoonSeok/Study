import numpy as np

a = np.array(range(1, 11))
size = 5

# 모델을 구성하시오

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)
dataset = split_x(a, size)
print("==================")
print(dataset)

x = dataset[:,:4]  # dataset[행, 렬]  dataset[0:6, 0:4]
y = dataset[:,4]  # dataset[0:6, 4]
print(y.shape)
print(x)
x = x.reshape(x.shape[0],x.shape[1],1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(4,1))) #(행, 열, 몇개씩자르는지)
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(1))

model.summary() 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x, y, epochs=2000, batch_size=16, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=16)


x_pred = dataset[-1,1:]
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)
print(y_pred)

# loss: 3.3817e-05
# [[11.988593]]

# loss: 5.0491e-07
# [[11.975308]]

# loss: 7.0105e-05
# [[12.104516]]

# loss: 6.1808e-06 -> node값 높임
# [[12.018729]]

# loss: 1.8278e-05
# [[12.0377445]]

# loss: 8.0636e-06
# [[12.024921]]