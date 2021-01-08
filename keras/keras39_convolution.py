import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3, 3), padding='same', strides=(1,1), input_shape=(10, 10, 1))) 
# 패딩을 먼저 때리고 자른다.
# 커널사이즈가 늘어나도 패딩을 그만큼 씌워줘서 shape를 맞춰준다.
# stride는 자를때 몇칸 건너뛰고 자를것인가를 결정한다. 디폴트는 1. padding = same보다 강력하네
model.add(MaxPooling2D(pool_size=(2,2)))
# 이미지를 pool_size=(n1, n2) 크기의 구간 여러개로 나눠서 구간 당 가장 큰수만 추출해서 다시 조합
model.add(Conv2D(9, (2, 2), padding='valid')) # 줄임가능 ,  padding은 valid가 디폴트임
# model.add(Conv2D(9, (2, 3)))
# model.add(Conv2D(8, 2)) # (n, n) -> n으로 표현 가능
model.add(Flatten())
model.add(Dense(1))

model.summary() 
# Conv2D number_parameters = out_channels * (in_channels(filters) * kernel_h * kernel_w + 1)
# output number_parameters after flatten = output * (in_channels * kernel_h * kernel_w + 1)

