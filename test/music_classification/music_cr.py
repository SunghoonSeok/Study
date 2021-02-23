import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import librosa, IPython
import librosa.display as lplt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Nadam
from sklearn.metrics.pairwise import cosine_similarity

class Classifi_recommend:

    def __init__(self, pred_csv_path, music_num=5):

        self.seed = 12
        self.pred_csv_path = pred_csv_path
        self.music_num = music_num

    def classification(self):
        a = os.path.splitext(self.pred_csv_path)
        a = os.path.split(a[0])


        df = pd.read_csv('c:/data/music/3s_data.csv')
        pred = pd.read_csv(self.pred_csv_path)
        print("Dataset has",df.shape)
        print("Count of Positive and Negative samples")
        print(df.label.value_counts().reset_index())


        # map labels to index
        label_index = dict()
        index_label = dict()
        for i, x in enumerate(df.label.unique()):
            label_index[x] = i
            index_label[i] = x
        df.label = [label_index[l] for l in df.label]

        df_shuffle = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        cpred = pred.copy()
        # remove irrelevant columns
        df_shuffle.drop(['filename', 'length','tempo'], axis=1, inplace=True)
        cpred.drop(['filename', 'length','tempo'], axis=1, inplace=True)
        df_y = df_shuffle.pop('label')
        df_x = df_shuffle
        x_pred = cpred

        # split into train dev and test
        
        x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df_y, train_size=0.7, random_state=self.seed, stratify=df_y)
        x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=self.seed, stratify=y_val_test)

        
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
        x_pred = pd.DataFrame(scaler.transform(x_pred), columns=x_pred.columns)



        optimizer = Adam(lr=0.001)

        model2 = load_model('c:/data/music/checkpoint/checkpoint_notempo_0.2917.hdf5',compile=False)
        # model2 = load_model('c:/data/music/checkpoint/checkpoint2_notempo_0.2885.hdf5',compile=False)
        # model2 = load_model('c:/data/music/checkpoint/checkpoint_notempo_0.2596.hdf5',compile=False)
        model2.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

        test_loss, test_acc  = model2.evaluate(x_test, y_test, batch_size=64)
        print("\nThe test Loss is :",test_loss)
        print("The Best test Accuracy is :",test_acc*100)
        y_pred = model2.predict(x_pred)
        y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)


        y_label=[]
        for i in range(len(y_recovery)):
            temp = index_label[y_recovery[i][0]]
            y_label.append(temp)



        print(""+str(a[1])+" 는(은) 무슨 장르니?")
        print(""+str(a[1])+"는(은)",y_label,"장르입니다.")

        i= random.randrange(len(y_label))
        
        df_30 = pd.read_csv('c:/data/music/30s_data.csv')
        pred = pred.assign(label=y_label)
        ipred = pred.iloc[[i]]
        npred = pred.iloc[i]
        fname = npred['filename']

        df_30 = pd.concat([df_30,ipred])
        df_30.set_index('filename', inplace=True)
        labels = df_30[['label']]
        df_30 = df_30.drop(columns=['length','label'])

        scaler2 = MaxAbsScaler()
        scaler2.fit(df_30)
        df_30 = scaler2.transform(df_30)
        
        similarity = cosine_similarity(df_30)
        sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)

        def find_similar_songs(name, n=5):
            series = sim_df[name].sort_values(ascending=False)
            series = series.drop(name)
            print("\n***************************\n"+name+" 와(과) 비슷한 곡 추천해줘")
            print(""+name+" 와(과) 비슷한 곡 "+str(n)+"개의 list입니다.")
            print(series.head(n).to_frame(' '))
        find_similar_songs(fname, self.music_num)
