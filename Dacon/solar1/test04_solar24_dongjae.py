# 동재가 만든 
# csv파일 합치고 중간값을 선택하여 다시 csv파일 생성
# 이상치를 없애는 효과를 기대 -> 효과가 있다!

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

# for i in range(1,6) :
#     df1.add(globals()['df{}'.format(i)], axis=1)
# df = df1.iloc[:,1:]
# df_2 = df1.iloc[:,:1]
# df_3 = (df/5).round(2)
# df_3.insert(0,'id',df_2)
# df3.to_csv('../data/csv/0122_timeseries_scale10.csv', index = False)

x = []
for i in range(20,25):
    df = pd.read_csv(f'c:/data/test/solar/sample_submission{i}_check.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'c:/data/test/solar/sample_submission{i}_check.csv', index_col=0, header=0)
for i in range(7776):
    for j in range(9):
        a = []
        for k in range(5):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('c:/data/test/solar/sample_submission25_check.csv')  