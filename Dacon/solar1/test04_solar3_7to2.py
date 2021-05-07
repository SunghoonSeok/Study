# 7일 묶어서 2일 뽑는거만 코딩

import pandas as pd
import numpy as np


from pandas import read_csv
df = read_csv('c:/data/test/solar/train/train.csv', index_col=None, header=0)
submission = read_csv('c:/data/test/solar/sample_submission.csv', index_col=None, header=0)

data = df.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
data =np.load('c:/data/test/solar/train.npy')

data = data.reshape(1095, 48, 9)
print(len(data)) # 1095

'''
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

x, y = split_xy(data, timesteps_x, timesteps_y, feature_x, feature_y)
print(x.shape, y.shape) # (1087, 7, 48, 5) (1087, 2, 48)

y = y.reshape(1087, 96)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=0)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D, GRU, SimpleRNN

# 모델1
input1 = Input(shape=(7, 48, 5))
dense1 = Conv2D(512, 2, padding='same')(input1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(256, 2, padding='same')(dense1)
# dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(128, 2, padding='same')(dense1)
dense1 = Flatten()(dense1)
dense1 = Dense(128)(dense1)
outputs = Dense(96)(dense1)

model = Model(inputs=input1, outputs=outputs)

#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# 모델 9번 돌리기 
  
for l in range(9):
    cp = ModelCheckpoint(filepath = 'c:/data/test/solar/checkpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
    model.fit(x,y,epochs= 1000, validation_split=0.2, batch_size =8, callbacks = [es,cp,lr])

    c = []
    for i in range(81):
        testx = pd.read_csv('c:/data/test/solar/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        testx = testx.to_numpy()  
        testx = testx.reshape(1,7,48,6)
        testx = testx[:,:,:,:-1]
        y_pred = model.predict(testx)
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
submission.to_csv('c:/data/test/solar/sample_submission3.csv', index=False)
'''