# solar 14,15에서 생성된 체크포인트 로드하기

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum

# 필요 함수 정의
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

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

def preprocess_data(data):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['GHI', 'DHI', 'DNI', 'WS', 'RH', 'T','TARGET']]                          
    return temp.iloc[:, :]

def hahaha(a, x_train, y_train, x_val, y_val, x_test):
    x = []
    for q in quantiles:
        filepath_cp = f'c:/data/test/solar/checkpoint/solar_checkpoint7_time{i}-{a}-{q}.hdf5'
        model = load_model(filepath_cp, compile = False)
        pred = pd.DataFrame(model.predict(x_test).round(2))
        x.append(pred)
    df_temp = pd.concat(x, axis = 1)
    df_temp[df_temp<0] = 0
     
    return df_temp

# 1. 데이터
train = pd.read_csv('c:/data/test/solar/train/train.csv')
sub = pd.read_csv('c:/data/test/solar/sample_submission.csv')

data = train.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
data =np.load('c:/data/test/solar/train.npy')

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


x_test_data = []
for j in range(81):
    file_path = 'c:/data/test/solar/test/' + str(j) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    x_test_data.append(temp)
x_test_data = pd.concat(x_test_data)
print(x_test_data.shape) # (27216,7)
x_test_data = np.array(x_test_data)
x_test_data = x_test_data.reshape(81,7,48,7)
x_test_data = np.transpose(x_test_data, axes=(2,0,1,3))

b=[]
c=[]
x_test=[]
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

    x_test = x_test_data[i,:,:,:]
    x_test = x_test.reshape(567,7)
    x_test = scaler.transform(x_test)
    x_test = x_test.reshape(81,7,7)
    temp_day7 = hahaha(0, x_train, y1_train, x_val, y1_val, x_test)
    temp_day8 = hahaha(1, x_train, y2_train, x_val, y2_val, x_test)
    b.append(temp_day7)
    c.append(temp_day8)
day7 = pd.concat(b, axis = 0)
day8 = pd.concat(c, axis = 0)

day7 = day7.to_numpy()
day8 = day8.to_numpy()
day7 = day7.reshape(48,81,9)
day8 = day8.reshape(48,81,9)
day7 = np.transpose(day7, axes=(1,0,2))
day8 = np.transpose(day8, axes=(1,0,2))
day7 = day7.reshape(3888, 9)
day8 = day8.reshape(3888, 9)
sub.loc[sub.id.str.contains("Day7"), "q_0.1":] = day7.round(2)
sub.loc[sub.id.str.contains("Day8"), "q_0.1":] = day8.round(2)

sub.to_csv('c:/data/test/solar/sample_submission13_check.csv', index=False)        

