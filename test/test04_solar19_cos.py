# 해가 뜨는 시간을 통일하는 것이 아닌
# 각 날짜별 해 뜨는 시간 해지는 시간 나타내어 cos값 대입하고
# GHI 구하기

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum

# 필요 함수 정의
train = pd.read_csv('c:/data/test/solar/train/train.csv')
sub = pd.read_csv('c:/data/test/solar/sample_submission.csv')

# 코사인 컬럼을 만들기 위한 함수
def make_cos(dataframe): 
    dataframe /=dataframe # 데이터프레임을 데이터 프레임으로 나눈다. 원래값이 0이 아니라면 1이겠지?
    c = dataframe.dropna() # 0인 값들은 NaN이 나오니 그 부분을 걸러내고 cos값 대입해줄거임
    d = c.to_numpy()
    # 해가 중천에 뜬 곳을 0도로 잡고 지면을 절대값 90도로 잡아 각도를 잡고 cosine값 생성
    def into_cosine(seq):
        for i in range(len(seq)):
            if i < len(seq)/2:
                seq[i] = float((len(seq)-1)/2) - (i)
            if i >= len(seq)/2:
                seq[i] = seq[len(seq) - i - 1]
        seq = seq/ np.max(seq) * np.pi/2
        seq = np.cos(seq)
        return seq

    d = into_cosine(d)
    dataframe = dataframe.replace(to_replace = np.NaN, value = 0) # Nan이 떴던 곳은 0으로 대체
    dataframe.loc[dataframe['cos'] == 1] = d # cos컬럼에 값 대입
    return dataframe

# 데이터 열 선택
def preprocess_data(data, is_train = True):
    a = pd.DataFrame()
    # 일별로 나누어 생성후 합침, 그래야 해 뜨고 지는 시간을 일별로 계산 가능
    for i in range(int(len(data)/48)): 
        tmp = pd.DataFrame()
        tmp['cos'] = data.loc[i*48:(i+1)*48-1,'TARGET']
        tmp['cos'] = make_cos(tmp)
        a = pd.concat([a,tmp])
    data['cos'] = a
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['GHI','DHI','DNI','WS','RH','T','TARGET']]
    return temp.iloc[:, :]

def split_x(data, size):
    x = []
    for i in range(len(data)-size+1):
        subset = data[i : (i+size)]
        x.append([item for item in subset])
    print(type(x))
    return np.array(x)

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                      
    

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

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

def only_compile(a, x_train, y_train, x_val, y_val):
    
    for q in quantiles:
        print('Day'+str(i)+' ' +str(q)+'실행중입니다.')
        model = DaconModel()
        optimizer = Adam(lr=0.002)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer = optimizer, metrics = [lambda y,y_pred: quantile_loss(q,y,y_pred)])
        filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint_time{i}-{a}-{q}.hdf5'
        cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
        model.fit(x_train,y_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y_val),callbacks = [es,lr,cp])
        
    return 

# 1. 데이터
# 데이터를 시간별로 나누기 전에 cos값을 대입하여 GHI를 구해내고 다시 시간별로 나눔
data = train.copy()
data = preprocess_data(data, is_train=True)
print(data.shape)
data = data.values
data = data.reshape(1095, 48, 7)
data = np.transpose(data, axes=(1,0,2))
print(data.shape)
data = data.reshape(48*1095,7)
df = train.copy()
df = preprocess_data(df, is_train=True)
df.loc[:,:] = data
df.to_csv('c:/data/test/solar/train_trans.csv', index=False)
# 시간별 모델 따로 생성
train_trans = pd.read_csv('c:/data/test/solar/train_trans.csv')
train_data = train_trans.copy()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)


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
    
    epochs = 10000
    bs = 32
    only_compile(0, x_train, y1_train, x_val, y1_val)
    only_compile(1, x_train, y2_train, x_val, y2_val)
