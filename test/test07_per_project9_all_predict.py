from music_classification.make_csv import Song_to_csv
import pandas as pd
import numpy as np
import librosa
import os

predict = Song_to_csv('c:/data/music/predict/','c:/data/music/predict_csv.csv', is_train=False)

predict.make_csv()
predict.fill_csv()