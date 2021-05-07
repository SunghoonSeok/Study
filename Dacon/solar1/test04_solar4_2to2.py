# 2일 묶어서 2일 뽑는거만 코딩

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

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)
x = data[:,:,3:8]
y = data[:,:,8]
x = split_x(x, 2)
y = split_x(y, 2)

print(x.shape, y.shape) # (1087, 2, 48, 5) (1087, 2, 48)

x = x.reshape(1094, 2*48*5)
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x)
x = scaler1.transform(x)

x = x.reshape(1094, 2, 48, 5)
y = y.reshape(-1, 96)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def ConvModel():
    input2 = Input(shape=(2, 48, 5))
    dense2 = Conv2D(512, 2, padding='same')(input2)
    dense2 = Conv2D(256, 2, padding='same')(dense2)
    dense2 = Conv2D(128, 2, padding='same')(dense2)
    dense2 = Flatten()(dense2)
    middle1 = Dense(256)(dense2)
    middle1 = Dense(128)(middle1)
    outputs = Dense(96)(middle1)
    model = Model(inputs=input2, outputs=outputs)
    return model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=12, mode='auto')
lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 6, verbose = 1)

def 
for q in quantiles:
    model = ConvModel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    model.fit(x, y, validation_split=0.2, epochs=1000, callbacks=[early_stopping, lr])
    
    c = []
    for i in range(81):
        testx = pd.read_csv('c:/data/test/solar/test/%d.csv'%i)
        testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        testx = testx.to_numpy()  
        testx = testx.reshape(1,7,48,6)
        
        testx = testx[:,5:,:,:-1]
        testx = testx.reshape(1, 2*48*5)
        testx = scaler1.transform(testx)
        testx = testx.reshape(1, 2, 48, 5)
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
    submission.loc[:, "q_%d"%q] = c
# submission = submission.iloc[:,:-1]
submission.to_csv('c:/data/test/solar/sample_submission6.csv', index=False)


# #3. 컴파일 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# es = EarlyStopping(monitor = 'val_loss', patience = 20)
# lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10, verbose = 1)
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

# # 모델 9번 돌리기 
  
# for l in range(9):
#     cp = ModelCheckpoint(filepath = 'c:/data/test/solar/checkpoint/dacon%d.hdf5'%l,monitor='val_loss', save_best_only=True)
#     model.fit(x,y,epochs= 1000, validation_split=0.2, batch_size =8, callbacks = [es,cp,lr])

#     c = []
#     for i in range(81):
#         testx = pd.read_csv('c:/data/test/solar/test/%d.csv'%i)
#         testx.drop(['Hour','Minute','Day'], axis =1, inplace = True)
#         testx = testx.to_numpy()  
#         testx = testx.reshape(1,7,48,6)
        
#         testx = testx[:,5:,:,:-1]
#         y_pred = model.predict(testx)
#         y_pred = y_pred.reshape(2,48)
        
  
#         a = []
#         for j in range(2):
#             b = []
#             for k in range(48):
#                 b.append(y_pred[j,k])
#             a.append(b)   
#         c.append(a)
#     c = np.array(c)
#     c = c.reshape(81*2*48,)
#     submission.loc[:, "q_0.%d"%(l+1)] = c
#     # c = np.array(c) # (81, 2, 48)

# submission.to_csv('c:/data/test/solar/sample_submission4.csv', index=False)
