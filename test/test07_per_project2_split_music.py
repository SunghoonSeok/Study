import os
import librosa
import numpy as np
import librosa.display
import soundfile

list = os.listdir('c:/data/music/genres_original/')

for genre in list:
    for i in range(100):
        y, sr = librosa.load("c:/data/music/genres_original/"+genre+"/"+genre+".%05d.wav"%i)
        for j in range(10):
            try: 
                y2 = y[22050*(3*j):22050*3*(j+1)]
                soundfile.write("c:/data/music/genres_split/"+genre+"/"+genre+".%05d.%d.wav"%(i,j), y2, samplerate=sr)
            except:
                print(genre,i,j,'실패')