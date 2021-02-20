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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
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

df_y =df_y.values
df_x = df_x.values
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, ReLU, LeakyReLU, ELU
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD, RMSprop, Adadelta, Ftrl, Nadam
from tensorflow.keras.activations import elu,relu,selu,swish,tanh

#2. 모델
def build_model(drop=0.2, optimizer='adam', node=512, activation='relu'):
    activation = activation
    model = Sequential()
    model.add(Dense(node, activation=activation, input_shape=(58,)))
    model.add(Dropout(drop))
    model.add(Dense(256, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(128, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(11, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    return model

def create_hyperparameters():
    batches = [16,32,64,128]
    optimizers = ['adam', 'adamax','adadelta','sgd','rmsprop', 'nadam', 'adagrad']
    dropout = [0.2, 0.3, 0.4]
    node = [512, 1024]
    activation = ['relu','elu','selu','swish','tanh']
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "node" : node, "activation" : activation}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min', patience=45)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, mode='min')
modelpath = 'c:/data/music/checkpoint/checkpoint3_notempo_{val_loss:.4f}.hdf5'
mc = ModelCheckpoint(modelpath, monitor='val_loss',save_best_only=True, mode='min',verbose=1)

model2 = KerasClassifier(build_fn=build_model, epochs=1000, batch_size=32, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
kfold2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
i = 1
a=[]
for train_index, test_index in kfold.split(df_x, df_y):
    print(str(i)+'번째 kfold split')
    x_train, x_test = df_x[train_index], df_x[test_index]
    y_train, y_test = df_y[train_index], df_y[test_index]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    search = GridSearchCV(model2, hyperparameters, cv=kfold2)
    search.fit(x_train, y_train, fit_params={'callbacks':[es,rl,mc]})
    acc = search.score(x_test, y_test)
    print("score :", acc)
    print(search.best_params_)
    print(search.best_score_)
    a.append(acc)
    i += 1
print("scores 모음 : ", a)
    
