import numpy as np
import pandas as pd

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

data0 = read_csv('c:/data/test/0.csv', index_col=None, header=0)
data0 = df.values
print(data0.shape)
np.save('c:/data/test/solar/train.npy', arr=data0)
y_predict = model.predict(x_test)
