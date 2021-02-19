import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import librosa, IPython
import librosa.display as lplt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
seed = 12
np.random.seed(seed)

a = os.path.splitext("c:/data/music/predict_music/predict_csv.wav")
a = os.path.split(a[0])
print(a[1])


df = pd.read_csv('c:/data/music/3s_data.csv')
pred = pd.read_csv('c:/data/music/predict_music/predict_csv/'+str(a[1])+'.csv')
print("Dataset has",df.shape)
print("Count of Positive and Negative samples")
print(df.label.value_counts().reset_index())


# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)
df.label = [label_index[l] for l in df.label]

df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)
pred_shuffle = pred.sample(frac=1, random_state=seed).reset_index(drop=True)
# remove irrelevant columns
df_shuffle.drop(['filename', 'length','tempo'], axis=1, inplace=True)
pred_shuffle.drop(['filename', 'length','tempo'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_x = df_shuffle
x_pred = pred_shuffle

# split into train dev and test

x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df_y, train_size=0.7, random_state=seed, stratify=df_y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=seed, stratify=y_val_test)


scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
x_pred = pd.DataFrame(scaler.transform(x_pred), columns=x_pred.columns)

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN, ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Ftrl, Nadam


optimizer = Adam(lr=0.001)
model2 = load_model('c:/data/music/checkpoint/checkpoint_notempo_0.2917.hdf5',compile=False)
model2.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

test_loss, test_acc  = model2.evaluate(x_test, y_test, batch_size=64)
print("The test Loss is :",test_loss)
print("\nThe Best test Accuracy is :",test_acc*100)
y_pred = model2.predict(x_pred)
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
y_recovery = index_label[y_recovery[0][0]]


print(""+str(a[1])+" 는(은) 무슨 장르니?")
print(""+str(a[1])+"는(은)",y_recovery,"장르입니다.")


df_30 = pd.read_csv('c:/data/music/30s_data.csv')
pred = pred.assign(label=[y_recovery])
df_30 = pd.concat([df_30,pred])
df_30.set_index('filename', inplace=True)
labels = df_30[['label']]
df_30 = df_30.drop(columns=['length','label'])

# print(df_30.head())
# print(df_30.tail())

from sklearn.preprocessing import MaxAbsScaler, Normalizer
scaler2 = MaxAbsScaler()
scaler2.fit(df_30)
df_30 = scaler2.transform(df_30)


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(df_30)
sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)
# print(sim_df.head())

def find_similar_songs(name, n=5):
    series = sim_df[name].sort_values(ascending=False)
    series = series.drop(name)
    print("\n*******\n"+name+" 와(과) 비슷한 곡 추천해줘")
    print(""+name+" 와(과) 비슷한 곡 "+str(n)+"개의 list입니다.")
    print(series.head(n).to_frame('추천목록'))
find_similar_songs(str(a[1]))
