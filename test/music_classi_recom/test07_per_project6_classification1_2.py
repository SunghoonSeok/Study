import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import random
import librosa
import os
seed = 12
np.random.seed(seed)

a = os.path.splitext("c:/data/music/predict_music/방탄소년단-Dynamite.wav")
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

# remove irrelevant columns
df_shuffle.drop(['filename', 'length', 'tempo'], axis=1, inplace=True)
cpred = pred.drop(['filename', 'length','tempo'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_x = df_shuffle
x_pred = cpred


from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df_y, train_size=0.7, random_state=seed, stratify=df_y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=seed, stratify=y_val_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
# x_pred = pd.DataFrame(scaler.transform(x_pred), columns=x_pred.columns)

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Nadam

model = Sequential()
model.add(Dense(512, activation='elu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))

optimizer = Adam(lr=0.0005)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min', patience=60)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min')
modelpath = 'c:/data/music/checkpoint/checkpoint2_notempo_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(modelpath, monitor='val_loss',save_best_only=True, mode='min',verbose=1)


model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_val, y_val), callbacks=[es,rl,mc])
test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size=32)

print("The test Loss is :",test_loss)
print("\nThe Best test Accuracy is :",test_acc*100)
print(plotHistory(hist, 'Batch_size=128'))

y_pred = model.predict(x_pred)
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
y_recovery = index_label[y_recovery[0][0]]

print(""+str(a[1])+" 는(은) 무슨 장르니?")
print(""+str(a[1])+" 는(은)",y_recovery,"장르입니다.")


df_30 = pd.read_csv('c:/data/music/30s_data.csv')
cpred = cpred.assign(label=[y_recovery])
df_30 = pd.concat([df_30,cpred])
df_30.set_index('filename', inplace=True)
labels = df_30[['label']]
df_30 = df_30.drop(columns=['length','label'])

scaler2 = StandardScaler()
scaler2.fit(df_30)
df_30 = scaler2.transform(df_30)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(df_30)
sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)

def find_similar_songs(name, n=5):
    series = sim_df[name].sort_values(ascending=False)
    series = series.drop(name)
    print("\n*******\n"+name+" 와(과) 비슷한 곡 추천해줘")
    print(""+name+" 와(과) 비슷한 곡 "+str(n)+"개의 list입니다.")
    print(series.head(n).to_frame('추천목록'))
find_similar_songs(a[1])
