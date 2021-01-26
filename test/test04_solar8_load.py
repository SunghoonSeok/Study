# solar7 checkpoint 집어 넣기

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum


train = pd.read_csv('c:/data/test/solar/train/train.csv')
sub = pd.read_csv('c:/data/test/solar/sample_submission.csv')

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
print(df_train.shape) # (52464, 9)

df_test = []

for i in range(81):
    file_path = 'c:/data/test/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

x_test = pd.concat(df_test)
print(x_test.shape) # (3888,7)


from sklearn.model_selection import train_test_split
x_train1, x_val1, y_train1, y_val1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=32)
x_train2, x_val2, y_train2, y_val2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=32)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train1)
x_train1 = scaler.transform(x_train1)
x_val1 = scaler.transform(x_val1)
x_train2 = scaler.transform(x_train2)
x_val2 = scaler.transform(x_val2)
x_test = scaler.transform(x_test)

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

x_train1 = x_train1.reshape(x_train1.shape[0], 1, x_train1.shape[1])
x_train2 = x_train2.reshape(x_train2.shape[0], 1, x_train2.shape[1])
x_val1 = x_val1.reshape(x_val1.shape[0], 1, x_val1.shape[1])
x_val2 = x_val2.reshape(x_val2.shape[0], 1, x_val2.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])




# 3. 컴파일, 훈련
def DaconModel():
    model = Sequential()
    model.add(Conv1D(512, 2, padding='same', input_shape=(1, 7)))
    model.add(Conv1D(256,2, padding='same'))
    model.add(Conv1D(128,2, padding='same'))
    model.add(Conv1D(64,2, padding='same'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1))
    return model


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
# filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint_{a}-{q}.hdf5'
# cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

epochs = 1000
bs = 32

def hahaha(a, x_train, y_train, x_val, y_val, x_test):
    x = []
    for q in quantiles:
        optimizer = Adam(lr=0.008)
        filepath_cp = f'c:/data/test/solar/checkpoint/solar_checkpoint_{a}-{q}.hdf5'
        model = load_model(filepath_cp, compile = False)
        # model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer = optimizer, metrics = [lambda y,y_pred: quantile_loss(q,y,y_pred)])
        # filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint_{a}-{q}.hdf5'
        # cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
        # model.fit(x_train,y_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y_val),callbacks = [es,lr,cp])
        pred = pd.DataFrame(model.predict(x_test).round(2))
        x.append(pred)
    df_temp = pd.concat(x, axis = 1)
    df_temp[df_temp<0] = 0
    num_temp = df_temp.to_numpy()
    return num_temp
num_temp1 = hahaha(1, x_train1, y_train1, x_val1, y_val1, x_test)
num_temp2 = hahaha(2, x_train2, y_train2, x_val2, y_val2, x_test)

print(num_temp1.shape, num_temp2.shape) # (3888, 63) (3888, 63)

sub.loc[sub.id.str.contains("Day7"), "q_0.1":] = num_temp1.round(2)
sub.loc[sub.id.str.contains("Day8"), "q_0.1":] = num_temp2.round(2)
sub.to_csv('c:/data/test/solar/sample_submission7_check.csv', index=False)
