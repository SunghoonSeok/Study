import pandas as pd
import numpy as np

train = pd.read_csv('c:/data/music/features_30_sec.csv', header=None)
feature = train.iloc[0,:]
feature = feature.values.tolist()
feature = feature[1:]
print(feature)
# ['filename', 'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 
# 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
# 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 
# 'perceptr_mean', 'perceptr_var', 'tempo', 
# 'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 
# 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 
# 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 
# 'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var', 'mfcc16_mean', 'mfcc16_var', 
# 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var', 
# 'label']

# 한 곡에 대한 feature 총 정리
import librosa
y, sr = librosa.load("c:/data/music/genres_original/blues/blues.00000.wav")
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
print(zero_crossing_rate_mean, zero_crossing_rate_var)

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

# # MFCCs
# S = librosa.feature.melspectrogram(y, sr=sr)
# S_DB = librosa.amplitude_to_db(S, ref=np.max)
# D = np.abs(librosa.stft(y, n_fft=2048, win_length=2048, hop_length=512))
# mfcc = librosa.feature.mfcc(y,S = librosa.power_to_db(D) sr=sr, n_mfcc = 20)
# print(mfcc[0])
# mfcc1_mean = np.mean(mfcc[0])
# mfcc1_var = np.var(mfcc[0])
# print(mfcc1_mean,mfcc1_var)