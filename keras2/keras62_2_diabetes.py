# epochs 100 적용
# validation split, callbacks 적용
# early stopping 5
# reduce lr 3

# model check

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42)
#1. 데이터 / 전처리


#2. 모델
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
def build_model(drop=0.5, optimizer='adam', node1=256, node2=128, activation='relu'):
    optimizer=optimizer()
    activation = activation
    inputs = Input(shape=(13,), name='input')
    x = Dense(node1,activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2,activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation=activation, name='hidden3')(x)
    x = Dense(32, activation=activation, name='hidden4')(x)
    x = Dense(16, activation=activation, name='hidden5')(x)
    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')
    return model

def create_hyperparameters():
    batches = [16,32,64,128]
    optimizer = [RMSprop, Adam, Adadelta]
    dropout = [0.1, 0.2, 0.3]
    node1 = [64, 128, 256]
    node2 = [64, 128, 256]
    activation=['relu', 'linear', 'tanh', 'selu']
    
    return {"batch_size" : batches, "optimizer" : optimizer, "drop" : dropout, "node1": node1, 
    "node2": node2, "activation":activation}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

model2 = KerasRegressor(build_fn=build_model, verbose=1, epochs=5, validation_split=0.2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 10)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras62_1_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
search.fit(x_train, y_train, verbose=1,epochs=100,validation_split=0.2, callbacks=[es,lr,cp]) # epochs default 1
# 둘다 쓰면 search에 넣은걸로 먹힌다
# r2 = search.score(x_test, y_test)
y_pred = search.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("최종 스코어 :", r2)
print(search.best_params_) # 내가 선택한 파라미터들
# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해
print(search.best_score_)

# 최종 스코어 : 0.4105976979397943
# {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop'>, 'node2': 128, 'node1': 64, 'drop': 0.1, 'batch_size': 16, 'activation': 'relu'}
# -68.05081558227539

