# 다량의 mp3파일을 wav 파일로 바꾸기
import os
from pydub import AudioSegment

audio_files = os.listdir('c:\\users\\ai\\downloads\\ballad\\ballad')
len_audio=len(audio_files)
for i in range (len_audio):
    if os.path.splitext(audio_files[i])[1] == ".mp3":
       mp3_sound = AudioSegment.from_mp3('c:\\users\\ai\\downloads\\ballad\\ballad\\'+str(audio_files[i])+'')
       mp3_sound.export("c:/users/ai/downloads/balwav/%05d.wav"%i, format="wav", bitrate='128k')

# 100곡의 발라드, 1:00 ~ 1:30 30초로 자르기
import librosa
import numpy as np
import librosa.display
import soundfile
for i in range(100):
    try:
        y, sr = librosa.load("c:/users/ai/downloads/balwav/%05d.wav"%i)
        y2 = y[22050*60:22050*90]
        soundfile.write("c:/data/music/genres_original/ballad/ballad.%05d.wav"%i, y2, samplerate=sr)
    except:
        print('%05d'%i,'번 빠졌습니다.')

# # 빠진 번호 곡 채우기
# except_num = [55,58]
# for j in except_num:
#     try:
#         y, sr = librosa.load("c:/users/ai/downloads/balwav/%05d.wav"%(j+50))
#         y2 = y[22050*60:22050*90]
#         soundfile.write("c:/data/music/genres_original/ballad/ballad.%05d.wav"%j, y2, samplerate=sr)
#     except:
#         print('%05d'%j,'번 빠졌습니다.')

# 파일 생성유무, Sampling rate, time 확인
y, sr = librosa.load("c:/data/music/genres_original/ballad/ballad.00055.wav")
print(len(y))
print('Sampling rate (KHz): %d' % sr)
print('Audio length (seconds): %.2f' % (len(y) / sr))
# 661500
# Sampling rate (KHz): 22050
# Audio length (seconds): 30.00