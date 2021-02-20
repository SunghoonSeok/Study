from music_classification.make_csv import Song_to_csv
from music_classification.music_cr import Classifi_recommend

# predict song to csv
csv_path = 'c:/data/music/아이유-celebrity.csv'
predict = Song_to_csv('c:/data/music/predict_music/아이유-celebrity.wav',csv_path, is_train=False, in_folder=False)

predict.make_csv()
predict.fill_csv()

result = Classifi_recommend(csv_path, music_num=7)
result.classification()