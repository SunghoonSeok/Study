# 7일과 2일 같이 묶은걸 앙상블
# quantile을 하려 했지만 그냥 모델만 9번 돌린것뿐...

import pandas as pd
import numpy as np


from pandas import read_csv
df = read_csv('c:/data/test/solar/train/train.csv', index_col=None, header=0)
submission = read_csv('c:/data/test/solar/sample_submission2.csv', index_col=None, header=0)

data = df.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
data =np.load('c:/data/test/solar/train.npy')

data = data.reshape(1095, 48, 9)

def split_xy(dataset, timesteps_x, timesteps_y, feature_x, feature_y):
    x, y = list(), list()
    
    for i in range(len(data)):
        x_end_number = i + timesteps_x
        y_end_number = x_end_number + timesteps_y
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i : x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    x = x[:,:,:,3:feature_x]
    y = y[:,:,:,feature_y]
    return x, y

timesteps_x = 7
timesteps_y = 2
feature_x = -1
feature_y = -1

x1, y = split_xy(data, timesteps_x, timesteps_y, feature_x, feature_y)
print(x1.shape, y.shape) # (1087, 7, 48, 5) (1087, 2, 48)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)
x2 = data[7:,:,3:8]
y2 = data[7:,:,8]
x2 = split_x(x2, 2)
y2 = split_x(y2, 2)

print(x2.shape, y2.shape) # (1087, 2, 48, 5) (1087, 2, 48)


y = y.reshape(1087, 96)
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=0)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D, GRU, SimpleRNN
# inputs = Input(shape=(6, x.shape[2]))
# dense1 = Conv1D(1000, 2, padding='same', activation='relu')(inputs)
# dense1 = MaxPooling1D(pool_size=2)(dense1)
# dense1 = Conv1D(500, 2, activation='relu')(dense1)
# dense1 = Conv1D(400, 2,activation='relu')(dense1)
# dense1 = Flatten()(dense1)

# 모델1
input1 = Input(shape=(7, 48, 5))
dense1 = Conv2D(512, 2, padding='same')(input1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(256, 2, padding='same')(dense1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(128, 2, padding='same')(dense1)
dense1 = Conv2D(64, 2, padding='same')(dense1)
dense1 = Flatten()(dense1)

# 모델2
input2 = Input(shape=(2, 48, 5))
dense2 = Conv2D(512, 2, padding='same')(input2)
# dense2 = Dropout(0.2)(dense1)
dense2 = Conv2D(256, 2, padding='same')(dense2)
# dense2 = Dropout(0.2)(dense1)
dense2 = Conv2D(128, 2, padding='same')(dense2)
dense2 = Flatten()(dense2)


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(256)(merge1)
middle1 = Dense(128)(middle1)
outputs = Dense(96)(middle1)

model = Model(inputs=[input1,input2], outputs=outputs)


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# 모델 9번 돌리기 
  
for l in range(9):
    cp = ModelCheckpoint(filepath = 'c:/data/test/solar/checkpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
    model.fit([x1_train,x2_train],y_train,epochs= 1000, validation_split=0.2, batch_size =8, callbacks = [es,cp,lr])

    c = []
    for i in range(81):
        testx = pd.read_csv('c:/data/test/solar/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        testx = testx.to_numpy()  
        testx = testx.reshape(1,7,48,6)
        testx1 = testx[:,:,:,:-1]
        testx2 = testx[:,5:,:,:-1]
        y_pred = model.predict([testx1,testx2])
        y_pred = y_pred.reshape(2,48)
        
  
        a = []
        for j in range(2):
            b = []
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)   
        c.append(a)
    c = np.array(c)
    c = c.reshape(81*2*48,)
    submission.loc[:, "q_0.%d"%(l+1)] = c
    # c = np.array(c) # (81, 2, 48)

submission.to_csv('c:/data/test/solar/sample_submission2.csv', index=False)