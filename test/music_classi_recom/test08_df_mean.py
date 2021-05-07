import numpy as np
import pandas as pd
feature = ['length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 
'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 
'perceptr_mean', 'perceptr_var', 'tempo', 
'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 
'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 
'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 
'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 
'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']
df1 = pd.read_csv("c:/data/music/3s_data.csv")
df2 = pd.read_csv("c:/data/music/30s_data.csv")
df3 = pd.read_csv("c:/data/music/train_3s.csv")
df4 = pd.read_csv("c:/data/music/train_30s.csv")


df_mean = pd.DataFrame(columns=feature, index=['ballad','blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock'])
t=[]
df4.drop(['filename','label'], axis=1, inplace=True)
for i in range(11):
    temp = df4.iloc[100*i:100*(i+1),:]
    temp = temp.mean()
    temp = temp.values
    t.append(temp)
t = np.array(t)
print(t.shape)
df_mean.iloc[:,:]=t
print(df_mean)
df_mean.to_csv('c:/data/music/df4_mean_30s.csv')
