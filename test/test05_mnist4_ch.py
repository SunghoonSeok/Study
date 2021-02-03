import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from pandas import read_csv
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
sub = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)

train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

train2 = train2.values
test2 = test2.values

train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

train2 = train2/255.
test2 = test2/255.

image = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
image2 = ImageDataGenerator()

kfold = StratifiedKFold