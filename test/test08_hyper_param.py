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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
seed = 12
np.random.seed(seed)

a = os.path.splitext("c:/data/music/predict_music/미란이-VVS.wav")
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
# pred_shuffle = pred.sample(frac=1, random_state=seed).reset_index(drop=True)
# remove irrelevant columns
df_shuffle.drop(['filename', 'length', 'tempo'], axis=1, inplace=True)
pred.drop(['filename', 'length','tempo'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_x = df_shuffle
x_pred = pred

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, ReLU, LeakyReLU, ELU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Ftrl, Nadam
from tensorflow.keras.activations import elu,relu,selu,swish,tanh

#2. 모델
def build_model(drop=0.2, activation1='relu',activation2='relu',activation3='relu',activation4='relu'):
    activation1 = activation1
    activation2 = activation2
    activation3 = activation3
    activation4 = activation4
    model = Sequential()
    model.add(Dense(512, activation=activation1, input_shape=(56,)))
    model.add(Dropout(drop))
    model.add(Dense(256, activation=activation2))
    model.add(Dropout(drop))
    model.add(Dense(128, activation=activation3))
    model.add(Dropout(drop))
    model.add(Dense(64, activation=activation4))
    model.add(Dropout(drop))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.0005), metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [32,64,128]
    activation1 = ['relu','elu','selu','swish']
    activation2 = ['relu','elu','selu','swish']
    activation3 = ['relu','elu','selu','swish']
    activation4 = ['relu','elu','selu','swish']

    return {"batch_size" : batches, "activation1" : activation1,
    "activation2" : activation2, "activation3" : activation3, "activation4" : activation4}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min', patience=60)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, mode='min')
modelpath = 'c:/data/music/checkpoint/checkpoint3_notempo_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(modelpath, monitor='val_loss',save_best_only=True, mode='min',verbose=1)

model2 = KerasClassifier(build_fn=build_model, epochs=1000, batch_size=32, verbose=1)

from sklearn.model_selection import train_test_split
x_train, x_val_test, y_train, y_val_test = train_test_split(df_x, df_y, train_size=0.7, random_state=seed, stratify=df_y)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, train_size=0.66, random_state=seed, stratify=y_val_test)


scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)
search = RandomizedSearchCV(model2, hyperparameters, n_iter=50)
hist = search.fit(x_train, y_train, validation_data=(x_val,y_val), callbacks=[es,rl,mc])
acc = search.score(x_test, y_test)
print("score :", acc)
print(search.best_params_)
print(search.best_score_)
    
# score : 0.9430097937583923
# {'batch_size': 32, 'activation4': 'elu', 'activation3': 'relu', 'activation2': 'elu', 'activation1': 'selu'}
# 0.9202493548393249