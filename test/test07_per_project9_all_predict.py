from music_classification.make_csv import Song_to_csv
import pandas as pd
import numpy as np
import librosa
import os

# # 3s train song to csv
# train = Song_to_csv('c:/data/music/genres_split/', 'c:/data/music/train_3s.csv', is_train=True)
# train.make_csv()
# train.fill_csv()

# # 30s song to csv
# rec = Song_to_csv('c:/data/music/genres_original/', 'c:/data/music/train_30s.csv', is_train=True)
# rec.make_csv()
# rec.fill_csv()



# predict song to csv
predict = Song_to_csv('c:/data/music/metal/','c:/data/music/metal_csv.csv', is_train=False)

predict.make_csv()
predict.fill_csv()