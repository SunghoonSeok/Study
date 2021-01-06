'''
keras08 split4 val2의 코드에서 train+test 값을 각각 1.1 0.9로 만들어 보자.
어떤 일이 일어나는지 체크
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.3, shuffle=True)




