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
np.random.seed(seed)
print(seed)

a = os.path.splitext("c:/data/music/predict_music/아이유-마음을드려요.wav")
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
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
pred_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_x = df_shuffle
x_pred = pred_shuffle

# split into train dev and test
from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df_y, train_size=0.7, random_state=seed, stratify=df_y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=seed, stratify=y_val_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
x_pred = pd.DataFrame(scaler.transform(x_pred), columns=x_pred.columns)

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, GRU, SimpleRNN, ReLU, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Ftrl, Nadam

# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(11, activation='softmax'))

# optimizer = Adam(lr=0.002)
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss',mode='min', patience=40)
# rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, mode='min')
# modelpath = 'c:/data/music/checkpoint/checkpoint_{val_loss:.4f}-{val_accuracy:.4f}.hdf5'
# mc = ModelCheckpoint(modelpath, monitor='val_loss',save_best_only=True, mode='min',verbose=1)


# model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=128, epochs=1000, validation_data=(x_val, y_val), callbacks=[es,rl,mc])

from xgboost import XGBClassifier, XGBRegressor
model = XGBClassifier(n_estimators=1000, learning_rate=0.04,
                    tree_method='gpu_hist',predictor='gpu_predictor', reg_lambda=3, reg_alpha=2, max_depth=9
)

model.fit(x_train,y_train, verbose=1,eval_metric='mlogloss',
         eval_set=[(x_train,y_train),(x_val,y_val)],early_stopping_rounds=10
)

aaa = model.score(x_test,y_test)


print(model.feature_importances_)
print("acc :", aaa)
'''
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
new_data=[]
feature=[]
a = np.percentile(model.feature_importances_, q=25)

for i in range(len(dataset.data[0])):
    if model.feature_importances_[i] > a:
       new_data.append(df.iloc[:,i])
       feature.append(dataset.feature_names[i])

new_data = pd.concat(new_data, axis=1)

        
x2_train, x2_test, y2_train, y2_test = train_test_split(new_data, dataset.target, train_size=0.8, random_state=32)
model2 =XGBClassifier(n_jobs=-1)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("acc2 :", acc2)
print(new_data.shape)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model, feature_name, data):
    n_features = data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_name)
    plt.xlabel("Feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)
plot_feature_importances_dataset(model, dataset.feature_names, dataset.data)
# plot_feature_importances_dataset(model2, feature, new_data)
plt.show()


# test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size=128)
# print("The test Loss is :",test_loss)
# print("\nThe Best test Accuracy is :",test_acc*100)
y_pred = model.predict(x_pred)
print(y_pred)

# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
y_recovery = index_label[y_pred[0]]

print(""+str(a[1])+" 는(은) 무슨 장르니?")
print(""+str(a[1])+" 는(은)",y_recovery,"장르입니다.")


df_30 = pd.read_csv('c:/data/music/30s_data.csv')
pred = pred.assign(label=[y_recovery])
df_30 = pd.concat([df_30,pred])
df_30.set_index('filename', inplace=True)
labels = df_30[['label']]
df_30 = df_30.drop(columns=['length','label'])

# print(df_30.head())
# print(df_30.tail())
df_30 = scaler.transform(df_30)
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