import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN
from tensorflow.keras.backend import mean, maximum


train = pd.read_csv('c:/data/test/solar/train/train.csv')
sub = pd.read_csv('c:/data/test/solar/sample_submission.csv')

def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

def preprocess_data(data, is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['Hour', 'TARGET','GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
        def split_xy(temp, timesteps_x, timesteps_y, feature_x, feature_y):
            x, y = list(), list()
    
            for i in range(len(data)):
                x_end_number = i + timesteps_x
                y_end_number = x_end_number + timesteps_y
                if y_end_number > len(temp):
                    break
                tmp_x = temp[i : x_end_number]
                tmp_y = temp[x_end_number : y_end_number]
                x.append(tmp_x)
                y.append(tmp_y)
            x = np.array(x)
            y = np.array(y)
            x = x[:,:,:,3:feature_x]
            y = y[:,:,:,feature_y]
            return x, y
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET','GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
# 상관계수
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark', font_scale=1.2, font='Malgun Gothic') # , palette='pastel'
sns.color_palette('Paired',6)
sns.heatmap(data=df_train.corr(), square=True, annot=True, cbar=True)
plt.show()

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
    model.add(Conv1D(512, 2, padding='same', input_shape=(1, 8), activation='relu'))
    model.add(Conv1D(256,2, padding='same', activation='relu'))
    model.add(Conv1D(128,2, padding='same', activation='relu'))
    model.add(Conv1D(64,2, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
# filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint2_{a}-{q}.hdf5'
# cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

epochs = 1000
bs = 64

def hahaha(a, x_train, y_train, x_val, y_val, x_test):
    x = []
    for q in quantiles:
        model = DaconModel()
        optimizer = Adam(lr=0.008)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), optimizer = optimizer, metrics = [lambda y,y_pred: quantile_loss(q,y,y_pred)])
        filepath = f'c:/data/test/solar/checkpoint/solar_checkpoint2_{a}-{q}.hdf5'
        cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
        model.fit(x_train,y_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y_val),callbacks = [es,lr,cp])
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
sub.to_csv('c:/data/test/solar/sample_submission8.csv', index=False)

