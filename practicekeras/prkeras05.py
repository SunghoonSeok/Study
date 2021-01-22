import pandas as pd
import numpy as np
from numpy import transpose

from pandas import read_csv
df = read_csv('c:/data/test/solar/train/train.csv', index_col=None, header=0)
submission = read_csv('c:/data/test/solar/sample_submission.csv', index_col=None, header=0)

data = df.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
data =np.load('c:/data/test/solar/train.npy')

data = data.reshape(1095, 48, 9)
data = transpose(data, axes=(1,0,2))
print(data.shape)
data = data.reshape(48*1095,9)
df.loc[:,:] = data
df.to_csv('c:/data/test/solar/train_trans.csv', index=False)