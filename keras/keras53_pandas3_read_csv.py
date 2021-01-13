import numpy as np
import pandas as pd
from pandas import read_csv
df = read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) # header가 없으면 none이라 해준다.

print(df)
