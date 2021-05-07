

import pandas as pd
import numpy as np


from pandas import read_csv
train = read_csv('c:/data/test/solar/train/train.csv', index_col=None, header=0)
submission = read_csv('c:/data/test/solar/sample_submission.csv', index_col=None, header=0)

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

df_test = []

for i in range(81):
    file_path = 'c:/data/test/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
X_test.shape



from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

....

model.compile(loss=lambda y,pred: quantile_loss(0.5,y,pred), **param)


q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for q in q_lst:
  model = sequantial()
  model.add(Dense(10))
  model.add(Dense(1))   
  model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam')
  model.fit(x,y, epoch=300)