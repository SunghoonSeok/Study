import numpy as np

y1 = np.array([3,6,5,4,2])

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y1 = to_categorical(y1)

print("to_categorical")
print("result : \n", y1)
print("shape : ", y1.shape)

y2 = np.array([3,6,5,4,2])
from sklearn.preprocessing import OneHotEncoder

y2 = y2.reshape(-1,1)
ohencoder = OneHotEncoder()
ohencoder.fit(y2)
y2 = ohencoder.transform(y2).toarray()

print("\n")
print("\n")
print("OneHotEncoder")
print("result : \n", y2)
print("shape : ", y2.shape)