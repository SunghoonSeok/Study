import numpy as np
a = np.mgrid[0:7, 0:7].T.reshape(-1,2)
print(a, a.shape) # (49,2)