import numpy as np
import pandas as pd

data = np.load('c:/data/test/samsung_jusik_all.npy')
print(data.shape) # (2399, 15)
# x를 5일치 y를 2일치로 잡아서 자르는 함수 만들자
# x의 특성은 14개 y의 특성은 5개


def split_xy(dataset, timesteps_x, timesteps_y, feature_x, feature_y):
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
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    x = x[:,:,:feature_x]
    y = y[:,:,:feature_y]
    return x, y

timesteps_x = 5
timesteps_y =2 
feature_x = 14
feature_y = 5
x, y = split_xy(data, timesteps_x, timesteps_y, feature_x, feature_y)
print(x.shape, y.shape) # (2393, 5, 14) (2393, 2, 5)