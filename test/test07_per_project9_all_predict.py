from music_classification.make_csv import Song_to_csv
import pandas as pd
import numpy as np
import librosa
import os
from music_classification.music_cr import Classifi_recommend
# # 3s train song to csv
# train = Song_to_csv('c:/data/music/genres_split/', 'c:/data/music/train_3s.csv', is_train=True)
# train.make_csv()
# train.fill_csv()

# # 30s song to csv
# rec = Song_to_csv('c:/data/music/genres_original/', 'c:/data/music/train_30s.csv', is_train=True)
# rec.make_csv()
# rec.fill_csv()



# predict song to csv
csv_path = 'c:/data/music/아이유-celebrity.csv'
predict = Song_to_csv('c:/data/music/predict_music/아이유-celebrity.wav',csv_path, is_train=False, in_folder=False)

predict.make_csv()
predict.fill_csv()

result = Classifi_recommend(csv_path, music_num=7)
result.classification()