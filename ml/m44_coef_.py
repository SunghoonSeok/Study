x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]


import matplotlib.pyplot as plt
plt.plot(x,y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X':x,'Y':y})
print(df)
print(df.shape) # (10, 2)

x_train = df.loc[:,'X']
y_train = df.loc[:,'Y']
print(x_train.shape, y_train.shape) # (10,) (10,)
print(type(x_train))

x_train = x_train.values.reshape(len(x_train),1)
print(x_train.shape, y_train.shape) # (10, 1) (10,)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print("Score :",score)

print("Weight :",model.coef_)
print("Bias :", model.intercept_)
# Score : 1.0
# Weight : [1.]
# Bias : 1.0