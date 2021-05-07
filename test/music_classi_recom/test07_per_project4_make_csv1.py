import pandas as pd
import numpy as np
import librosa
import os

feature = ['length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 
'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 
'perceptr_mean', 'perceptr_var', 'tempo', 
'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 
'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 
'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 
'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 
'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 
'label']

list = os.listdir('c:/data/music/genres_original/')
a =[]
for genre in list:
    index = os.listdir('c:/data/music/genres_original/'+genre+'')
    for i in range(100):
        a.append(index[i])

df_30 = pd.DataFrame(columns=feature, index=a)
df_30.index.name = 'filename'
print(df_30.head())
print(df_30.tail())
pd.DataFrame.to_csv(df_30, 'c:/data/music/df_30.csv')

