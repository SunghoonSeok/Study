import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats
# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)

x = []
for i in range(6,23):
    df = pd.read_csv(f'../data/lotte2/answer ({i}).csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

a = []
df = pd.read_csv(f'../data/lotte2/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        for k in range(17):
            a.append(x[k,i,j].astype('int'))
a = np.array(a)
a = a.reshape(72000,17)
m = []
for i in range (72000) :  
    b = stats.mode(a[i])
    m.append(b[0])

m = np.array(m)
print(m.shape)
sub = pd.read_csv('../data/lotte2/sample.csv')
sub['prediction'] = pd.DataFrame(m)
sub.to_csv('../data/lotte2/add_file18.csv',index=False)