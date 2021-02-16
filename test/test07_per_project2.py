import pandas as pd
import numpy as np

train = pd.read_csv('c:/data/music/features_30_sec.csv', header=None)
print(train.iloc[0,:])

feature = ['length', 'chroma', 'rms', 'spectral_centroid', 'spectral_bandwidth','rolloff'
          'zero_crossing_rate', 'harmony', 'perceptr', 'tempo', 'mfcc', 'label']