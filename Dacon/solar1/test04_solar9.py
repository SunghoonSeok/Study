# 7일의 데이터로 2일의 target값 구하기
# 시간별로 데이터를 나눠서 훈련

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum

# 필요 함수 정의

# GHI추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# 데이터 몇일씩 자르는 함수
def split_x(data, size):
    x = []
    for i in range(len(data)-size+1):
        subset = data[i : (i+size)]
        x.append([item for item in subset])
    print(type(x))
    return np.array(x)

# quantile loss 관련 함수
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 데이터 컬럼을 7개만 쓰겠다
def preprocess_data(data):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T','TARGET']]                          
    return temp.iloc[:, :]

# 모델, Conv1D사용
def DaconModel():
    model = Sequential()
    model.add(Conv1D(256,2, padding='same', input_shape=(7, 7),activation='relu'))
    model.add(Conv1D(128,2, padding='same',activation='relu'))
    model.add(Conv1D(64,2, padding='same',activation='relu'))
    model.add(Conv1D(32,2, padding='same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1))
    return model

# optimizer 불러오기
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# 컴파일 훈련 함수, optimizer 변수처리하여 lr=0.002부터 줄여나가도록 한다
# lr을 for문 밖에 두면 초기화가 되지 않으니 명심할것
# 총 48(시간수)*9(quantile)*2(Day7,8)개의 체크포인트모델이 생성됨
def only_compile(a, x_train, y_train, x_val, y_val):
    
    for q in quantiles:
        print('Day'+str(i)+' ' +str(q)+'실행중입니다.')
        model = DaconModel()
        optimizer = Adam(lr=0.002)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer = optimizer, metrics = [lambda y,y_pred: quantile_loss(q,y,y_pred)])
        filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint5_time{i}-{a}-{q}.hdf5'
        cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
        model.fit(x_train,y_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y_val),callbacks = [es,lr,cp])
        
    return 



# 1. 데이터
train = pd.read_csv('c:/data/test/solar/train/train.csv')
sub = pd.read_csv('c:/data/test/solar/sample_submission.csv')

# 데이터 npy로 바꾸기
data = train.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
data =np.load('c:/data/test/solar/train.npy')

# 전치를 활용한 데이터 시간별 묶음
data = data.reshape(1095, 48, 9)
data = np.transpose(data, axes=(1,0,2))
print(data.shape)
data = data.reshape(48*1095,9)
df = train.copy()
df.loc[:,:] = data
df.to_csv('c:/data/test/solar/train_trans.csv', index=False)
# 시간별 모델 따로 생성
train_trans = pd.read_csv('c:/data/test/solar/train_trans.csv')
train_data = preprocess_data(train_trans) # (52560,7)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 15)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)

# for문으로 시간, quantile, day7,8 을 구분하여 체크포인트 생성
for i in range(48):
    train_sort = train_data[1095*(i):1095*(i+1)]
    train_sort = np.array(train_sort)
    y = train_sort[7:,-1] #(1088,)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_sort)
    train_sort = scaler.transform(train_sort)

    x = split_x(train_sort, 7)
    x = x[:-2,:] #(1087,7,7)
    y1 = y[:-1] #(1087,)
    y2 = y[1:] #(1087,)

    from sklearn.model_selection import train_test_split
    x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x, y1, y2, train_size=0.8, shuffle=True, random_state=32)
    
    epochs = 1000
    bs = 32
    only_compile(0, x_train, y1_train, x_val, y1_val)
    only_compile(1, x_train, y2_train, x_val, y2_val)


    



