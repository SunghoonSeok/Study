# feature = ['length', 'chroma', 'rms', 'spectral_centroid', 'spectral_bandwidth','rolloff'
#           'zero_crossing_rate', 'harmony', 'perceptr', 'tempo', 'mfcc', 'label']

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

y, sr = librosa.load('c:/users/ai/documents/gomrecorder/정승환_눈사람.wav')

print(y)
print(len(y))
print('Sampling rate (KHz): %d' % sr)
print('Audio length (seconds): %.2f' % (len(y) / sr))

# [0.10967798 0.127714   0.12182648 ... 0.02821436 0.02284489 0.01139307]
# 661500
# Sampling rate (KHz): 22050
# Audio length (seconds): 30.00
audio_file, _ = librosa.effects.trim(y)

plt.figure(figsize = (16, 6))
librosa.display.waveplot(y = y, sr = sr, color = "#A300F9");
plt.title("Sound Waves in Snowman", fontsize = 23);
# plt.show()

# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns

# Short-time Fourier transform (STFT)
D = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length))

print('Shape of D object:', np.shape(D)) # Shape of D object: (1025, 1292)
# shape = (1 + n_fft/2, len(y)/hop_length)
plt.figure(figsize = (16, 6))
plt.plot(D);
plt.show()


# Convert an amplitude spectrogram to Decibels-scaled spectrogram.
D = np.abs(librosa.stft(y, n_fft = n_fft, hop_length = hop_length))
DB = librosa.amplitude_to_db(D, ref = np.max)

# Creating the Spectogram
plt.figure(figsize = (16, 6))
librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log')
plt.colorbar();
# plt.show()


y, _ = librosa.effects.trim(y)


S = librosa.feature.melspectrogram(y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(16, 6))
librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
plt.colorbar()
# plt.show()

# Total zero_crossings in our 1 song
zero_crossings = librosa.zero_crossings(audio_file, pad=False)
print(sum(zero_crossings)) # 50603

y_harm, y_perc = librosa.effects.hpss(audio_file)
print(y_harm)
print(y_perc)

plt.figure(figsize = (16, 6))
plt.plot(y_harm, color = '#A300F9');
plt.plot(y_perc, color = '#FFB100');
# plt.show()

tempo, _ = librosa.beat.beat_track(y, sr = sr)
print(tempo) # 184.5703125

import sklearn
def normalize(x, axis=0):
  return sklearn.preprocessing.minmax_scale(x, axis=axis)

mfccs = librosa.feature.mfcc(y, sr=sr)
mfccs = normalize(mfccs, axis=1)

print(mfccs)
print(mfccs.shape)

plt.figure(figsize=(16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()
'''