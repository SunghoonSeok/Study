# mp3파일을 wav 파일로 바꾸기
import os
from pydub import AudioSegment


mp3_sound = AudioSegment.from_mp3('c:\\data\\music\\country\\country\\countrycountry.mp3')
mp3_sound.export('c:\\data\\music\\country\\country\\countrycountry2.wav', format="wav")

# 통으로 되어있는 곡 4분마다 30초씩
import librosa
import numpy as np
import librosa.display
import soundfile
y, sr = librosa.load('c:\\data\\music\\country\\country\\countrycountry2.wav')
for i in range(100):
    y2 = y[22050*240*(i):22050*(240*(i)+30)]
    soundfile.write("c:/data/music/country/country/country.%05d.wav"%i, y2, samplerate=sr)


# # 빠진 번호 곡 채우기
# except_num = [55,58]
# for j in except_num:
#     try:
#         y, sr = librosa.load("c:/users/ai/downloads/balwav/%05d.wav"%(j+50))
#         y2 = y[22050*60:22050*90]
#         soundfile.write("c:/data/music/genres_original/ballad/ballad.%05d.wav"%j, y2, samplerate=sr)
#     except:
#         print('%05d'%j,'번 빠졌습니다.')

# # 파일 생성유무, Sampling rate, time 확인
# y, sr = librosa.load("c:/data/music/genres_original/ballad/ballad.00055.wav")
# print(len(y))
# print('Sampling rate (KHz): %d' % sr)
# print('Audio length (seconds): %.2f' % (len(y) / sr))
# # 661500
# # Sampling rate (KHz): 22050
# # Audio length (seconds): 30.00