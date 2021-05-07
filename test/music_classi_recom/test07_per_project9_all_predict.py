from music_classification.make_csv import Song_to_csv
from music_classification.music_cr import Classifi_recommend
# 3s train song to csv
train = Song_to_csv('c:/data/music/genres_split/', 'c:/data/music/3s_data.csv', is_train=True, in_folder=True)
train.make_csv()
train.fill_csv()

# 30s song to csv
recommend = Song_to_csv('c:/data/music/genres_original/', 'c:/data/music/30s_data.csv', is_train=True, in_folder=True)
recommend.make_csv()
recommend.fill_csv()


# predict song to csv
csv_path = 'c:/data/music/predict_csv.csv'
predict = Song_to_csv('c:/data/music/predict/',csv_path, is_train=False, in_folder=True)

predict.make_csv()
predict.fill_csv()
