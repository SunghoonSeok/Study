# 실습
# outliers1 을 행렬형태도 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
               [1000,2000,3,4000,5000,-100000,7000,8,9000,10000]])
aaa = aaa.transpose()
print(aaa.shape) # (10, 2)


def outliers(data_out):
    for i in range(data_out.shape[-1]):
        data = data_out[:,i]
        quartile_1, q2, quartile_3 = np.percentile(data, [25, 50, 75])
        print("======= feature %d ======"%i)
        print("1사분위 :",quartile_1)
        print("q2 :",q2)
        print("3사분위 :", quartile_3)
        iqr = quartile_3 - quartile_1
        print("iqr :", iqr)
        lower_bound = quartile_1 - (iqr*1.5)
        upper_bound = quartile_3 + (iqr*1.5)
        print("이상치의 위치 :", np.where((data>upper_bound) | (data<lower_bound)))
        print("=========================")
        
outliers(aaa)

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()