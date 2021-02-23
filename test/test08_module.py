from music_classification.make_csv import Song_to_csv
from music_classification.music_cr import Classifi_recommend

# predict song to csv
csv_path = 'c:/data/music/disco_csv.csv'
predict = Song_to_csv('c:/data/music/disco/',csv_path, is_train=False, in_folder=True)

# predict.make_csv()
# predict.fill_csv()

result = Classifi_recommend(csv_path, music_num=7)
result.classification()