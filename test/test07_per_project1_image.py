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



# [-0.15219471 -0.18531479 -0.17492233 ...  0.12399548  0.08108243
#   0.03574109]
# 661500
# Sampling rate (KHz): 22050
# Audio length (seconds): 30.00

audio_file, _ = librosa.effects.trim(y)

# the result is an numpy ndarray
print('Audio File:', audio_file, '\n')
print('Audio File shape:', np.shape(audio_file))

plt.figure(figsize = (16, 6))
librosa.display.waveplot(y = audio_file, sr = sr, color = "#A300F9");
plt.title("Sound Waves in IU-I give you my heart", fontsize = 23);
# plt.show()

# Default FFT window size
n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

# Short-time Fourier transform (STFT)
D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))

print('Shape of D object:', np.shape(D)) # Shape of D object: (1025, 1292)

plt.figure(figsize = (16, 6))
plt.plot(D);
# plt.show()

# Convert an amplitude spectrogram to Decibels-scaled spectrogram.
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

