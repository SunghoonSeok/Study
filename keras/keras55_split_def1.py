import numpy as np
import pandas as pd

data = np.load('c:/data/test/samsung_jusik_all.npy')
print(data.shape) # (2399, 15)
# x를 5일치 y를 2일치로 잡아서 자르는 함수 만들자


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


timesteps_x = 5
timesteps_y =2 
x, y = split_xy(data, timesteps_x, timesteps_y)
print(x.shape, y.shape) # (2393, 5, 15) (2393, 2, 15)