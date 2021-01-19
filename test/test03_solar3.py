import numpy as np
import pandas as pd
import os
import glob
import random

data = np.load('c:/data/test/solar/train.npy')
print(data.shape)
data = data.reshape(1095, 48, 9)

def split_xy(dataset, timesteps_x, timesteps_y):
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
    return np.array(x), np.array(y)

timesteps_x = 7
timesteps_y = 2

x, y = split_xy(data, timesteps_x, timesteps_y)
print(x.shape, y.shape) # (1087, 7, 48, 9) (1087, 2, 48, 9)
x = x[:,:,:,3:]
y = y[:,:,:,3:]
print(x.shape, y.shape) # (1087, 7, 48, 6) (1087, 2, 48, 6)

y = y.reshape(1087, 2*48*6)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Flatten
model = load_model('c:/data/test/solar/solar_model.h5')



loss, mae = model.evaluate(x_test, y_test, batch_size=64)

y_pred = []

for i in range(81) :
    filepath = '../data/test/solar/test/{}.csv'.format(i)
    globals()['pred{}'.format(i)] = pd.read_csv(filepath,index_col=False)
    globals()['pred_{}'.format(i)] = globals()['pred{}'.format(i)].iloc[:,3:]
    globals()['pred_{}'.format(i)] = globals()['pred_{}'.format(i)].to_numpy()
    globals()['pred_{}'.format(i)] = (globals()['pred_{}'.format(i)]).reshape(1,7, 48, 6)
    globals()['y_pred_{}'.format(i)] = model.predict(globals()['pred_{}'.format(i)])
    y_pred.append(globals()['y_pred_{}'.format(i)])
y_pred = np.array(y_pred)
y_pred = y_pred.reshape(81*96, 6)
y_pred = np.round_(y_pred, 2)
print(y_pred)


df2 = pd.DataFrame(y_pred)
print(df2)
df2.to_csv('c:/data/sample_submission.csv', sep=',')
'''
from pandas import read_csv
df = read_csv('c:/data/test/sample_submission.csv', index_col=0, header=0)
print(df)

print(df.shape, df2.shape)
print(df2.loc[:,0:])

df.loc[:, 'q_0.4':] = df2.loc[:,0:]
print(df)
'''

'''
# y = y.reshape(y.shape[0],96,6)
# for b in range(81) :
#     globals()['pred{}'.format(b)]= (globals()['pred{}'.format(b)]).reshape(1,336,6)
result = model.predict(pred0)
result = result.reshape(-1,96,6)
print("\n",result)
'''