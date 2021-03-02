import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.activations import elu, selu
def selu(x, alpha=1, scale=1.0507):
    return scale*x if x>=0 else scale*alpha*(np.exp(x)-1)

x = np.arange(-5,5,0.1)
print(x)
y=[]
for i in x:
    a = selu(i,1.3)
    y.append(a)
print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()

#### 과제
# elu, selu, reaky relu
# 72_2,3,4번으로 파일을 만들것