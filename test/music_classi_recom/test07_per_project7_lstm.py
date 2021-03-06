import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import librosa, IPython
import librosa.display as lplt
import os
seed = 12
seed = np.random.seed(seed)
print(seed)

a = os.path.splitext("c:/data/music/predict_music/미란이-VVS.wav")
a = os.path.split(a[0])
print(a[1])


df = pd.read_csv('c:/data/music/train_3s.csv')
df2 = pd.read_csv('c:/data/music/train_30s.csv')
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
df2.label = [label_index[l] for l in df2.label]


# df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)
# pred_shuffle = pred.sample(frac=1, random_state=seed).reset_index(drop=True)
# # remove irrelevant columns
df.drop(['filename', 'length', 'tempo'], axis=1, inplace=True)
pred.drop(['filename', 'length','tempo'], axis=1, inplace=True)
df_y = df.pop('label')
df_x = df
x_pred = pred
df2_y = df2.pop('label')
df_x = df_x.values
x_pred = x_pred.values

print(df_x.shape)
df_x = df_x.reshape(1100, 10, 56)



# split into train dev and test
from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df2_y, train_size=0.7, random_state=seed, stratify=df2_y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=seed, stratify=y_val_test)

print(x_train.shape, x_val.shape, x_test.shape) # (770, 10, 56) (217, 10, 56) (113, 10, 56)
x_train = x_train.reshape(7700,56)
x_val = x_val.reshape(2170,56)
x_test = x_test.reshape(1130,56)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train =scaler.transform(x_train)
x_val =scaler.transform(x_val)
x_test =scaler.transform(x_test)

x_train = x_train.reshape(770, 10, 56)
x_val = x_val.reshape(217,10,56)
x_test = x_test.reshape(113,10,56)



from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN, ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Ftrl, Nadam

model = Sequential()
model.add(LSTM(1024, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))

optimizer = Adam(lr=0.001)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min', patience=90)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, mode='min')
# modelpath = 'c:/data/music/checkpoint/checkpoint_{val_loss:.4f}-{val_accuracy:.4f}.hdf5'
modelpath = 'c:/data/music/checkpoint/checkpoint_notempo_{val_loss:.4f}_lstm.hdf5'
mc = ModelCheckpoint(modelpath, monitor='val_loss',save_best_only=True, mode='min',verbose=1)


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=1000, validation_data=(x_val, y_val), callbacks=[es,rl,mc])


test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size=64)

print("The test Loss is :",test_loss)
print("\nThe Best test Accuracy is :",test_acc*100)
'''
y_pred = model.predict(x_pred)
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
y_recovery = index_label[y_recovery[0][0]]


# model2 = load_model('c:/data/music/checkpoint/checkpoint.hdf5')
# test_loss, test_acc  = model2.evaluate(x_test, y_test, batch_size=128)
# print("The test Loss is :",test_loss)
# print("\nThe Best test Accuracy is :",test_acc*100)
# y_pred = model2.predict(x_pred)
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# y_recovery = index_label[y_recovery[0][0]]


print(""+str(a[1])+" 는(은) 무슨 장르니?")
print(""+str(a[1])+" 는(은)",y_recovery,"장르입니다.")


df_30 = pd.read_csv('c:/data/music/train_30s.csv')
pred = pred.assign(label=[y_recovery])
df_30 = pd.concat([df_30,pred])
df_30.set_index('filename', inplace=True)
labels = df_30[['label']]
df_30 = df_30.drop(columns=['length','label'])

scaler2 = StandardScaler()
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
find_similar_songs(a[1])
'''