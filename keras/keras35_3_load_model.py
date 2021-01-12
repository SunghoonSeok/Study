import numpy as np
a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("==================")
print(dataset)
x = dataset[:,:4]
y = dataset[:,4] 
x = x.reshape(x.shape[0],x.shape[1],1)

# 2. 모델 구성
from tensorflow.keras.models import load_model
model = load_model("../data/h5/save_keras35.h5")

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x, y, epochs=2000, batch_size=8, callbacks=[early_stopping])



# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=8)


x_pred = dataset[-1,1:]
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)
print(y_pred)

