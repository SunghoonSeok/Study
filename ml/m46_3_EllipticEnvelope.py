from sklearn.covariance import EllipticEnvelope
import numpy as np
aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000],
               [1000,2000,3,4000,5000,-100000,7000,8,9000,10000,110]])
aaa = np.transpose(aaa)
print(aaa.shape)

outlier = EllipticEnvelope(contamination=.3)
outlier.fit(aaa)

print(outlier.predict(aaa))