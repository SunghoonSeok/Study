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
'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']

a = os.path.splitext("c:/data/music/predict_music/방탄소년단-Dynamite.wav")
a = os.path.split(a[0])

df_pred = pd.DataFrame(columns=feature, index=[a[1]])
df_pred.index.name = 'filename'
pd.DataFrame.to_csv(df_pred, 'c:/data/music/predict_music/predict_csv/'+str(a[1])+'.csv')

import pandas as pd
import numpy as np



# 한 곡에 대한 feature 총 정리
import librosa
y, sr = librosa.load("c:/data/music/predict_music/"+str(a[1])+".wav")
# from playsound import playsound
# playsound("c:/data/music/genres_original/ballad/ballad.00000.wav")

# length
length = len(y) #661500

# chroma_stft
chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)
chroma_stft_mean = np.mean(chroma_stft) #0.29205593
chroma_stft_var = np.var(chroma_stft) #0.09444008

# rms
rms = librosa.feature.rms(y)
rms_mean = np.mean(rms)
rms_var = np.var(rms)

# spectral_centroid
spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)
spectral_centroid_mean = np.mean(spectral_centroids)
spectral_centroid_var = np.var(spectral_centroids)

# spectral_bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
spectral_bandwidth_mean = np.mean(spectral_bandwidth)
spectral_bandwidth_var = np.var(spectral_bandwidth)

# rolloff
rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
rolloff_mean = np.mean(rolloff)
rolloff_var = np.var(rolloff)

# zero_crossing_rate ?

zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
zero_crossing_rate_mean = np.mean(zero_crossing_rate)
zero_crossing_rate_var = np.var(zero_crossing_rate)
# print(zero_crossing_rate_mean, zero_crossing_rate_var)

# harmony, perceptr
harmony, perceptr = librosa.effects.hpss(y)
harmony_mean = np.mean(harmony)
harmony_var = np.var(harmony)
perceptr_mean = np.mean(perceptr)
perceptr_var = np.var(perceptr)
# print(harmony_mean,harmony_var,perceptr_mean,perceptr_var)

# tempo
tempo, _ = librosa.beat.beat_track(y, sr=sr)
# print(tempo)

# MFCCs
S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
D = np.abs(librosa.stft(y, n_fft=2048, win_length=2048, hop_length=512))
mfcc = librosa.feature.mfcc(y, sr=sr, S=librosa.power_to_db(D), n_mfcc = 20)
print(np.mean(mfcc[0]))  # -11.3112

mfcc1_mean = np.mean(mfcc[0])
mfcc1_var = np.var(mfcc[0])
mfcc2_mean = np.mean(mfcc[1])
mfcc2_var = np.var(mfcc[1])
mfcc3_mean = np.mean(mfcc[2])
mfcc3_var = np.var(mfcc[2])
mfcc4_mean = np.mean(mfcc[3])
mfcc4_var = np.var(mfcc[3])
mfcc5_mean = np.mean(mfcc[4])
mfcc5_var = np.var(mfcc[4])
mfcc6_mean = np.mean(mfcc[5])
mfcc6_var = np.var(mfcc[5])
mfcc7_mean = np.mean(mfcc[6])
mfcc7_var = np.var(mfcc[6])
mfcc8_mean = np.mean(mfcc[7])
mfcc8_var = np.var(mfcc[7])
mfcc9_mean = np.mean(mfcc[8])
mfcc9_var = np.var(mfcc[8])
mfcc10_mean = np.mean(mfcc[9])
mfcc10_var = np.var(mfcc[9])
mfcc11_mean = np.mean(mfcc[10])
mfcc11_var = np.var(mfcc[10])
mfcc12_mean = np.mean(mfcc[11])
mfcc12_var = np.var(mfcc[11])
mfcc13_mean = np.mean(mfcc[12])
mfcc13_var = np.var(mfcc[12])
mfcc14_mean = np.mean(mfcc[13])
mfcc14_var = np.var(mfcc[13])
mfcc15_mean = np.mean(mfcc[14])
mfcc15_var = np.var(mfcc[14])
mfcc16_mean = np.mean(mfcc[15])
mfcc16_var = np.var(mfcc[15])
mfcc17_mean = np.mean(mfcc[16])
mfcc17_var = np.var(mfcc[16])
mfcc18_mean = np.mean(mfcc[17])
mfcc18_var = np.var(mfcc[17])
mfcc19_mean = np.mean(mfcc[18])
mfcc19_var = np.var(mfcc[18])
mfcc20_mean = np.mean(mfcc[19])
mfcc20_var = np.var(mfcc[19])

feature =[length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, 
spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, 
rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var, 
perceptr_mean, perceptr_var, tempo, 
mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, 
mfcc5_mean, mfcc5_var, mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, 
mfcc9_mean, mfcc9_var, mfcc10_mean, mfcc10_var, mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var, 
mfcc13_mean, mfcc13_var, mfcc14_mean, mfcc14_var, mfcc15_mean, mfcc15_var, mfcc16_mean, mfcc16_var, 
mfcc17_mean, mfcc17_var, mfcc18_mean, mfcc18_var, mfcc19_mean, mfcc19_var, mfcc20_mean, mfcc20_var]

import pandas as pd
pred = pd.read_csv('c:/data/music/predict_music/predict_csv/'+str(a[1])+'.csv', index_col=0, header=0)

pred.loc[:,:]=feature
pred.to_csv('c:/data/music/predict_music/predict_csv/'+str(a[1])+'2.csv')
