# epochs 100 적용
# validation split, callbacks 적용
# early stopping 5
# reduce lr 3

# model check

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
def build_model(drop=0.5, optimizer='adam', node1=256, node2=128, lr=0.001, activation='relu'):
    optimizer=optimizer(lr=lr)
    activation = activation
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(node1,activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2,activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation=activation, name='hidden3')(x)
    x = Dense(32, activation=activation, name='hidden4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [16,32,64,128]
    optimizer = [RMSprop, Adam, Adadelta]
    dropout = [0.1, 0.2, 0.3]
    node1 = [64, 128, 256]
    node2 = [64, 128, 256]
    lr=[0.01,0.005,0.001]
    activation=['relu', 'linear', 'tanh', 'sigmoid']
    
    return {"batch_size" : batches, "optimizer" : optimizer, "drop" : dropout, "node1": node1, 
    "node2": node2, "lr":lr, "activation":activation}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2 = KerasClassifier(build_fn=build_model, verbose=1, epochs=5, validation_split=0.2)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 5)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras61_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
search.fit(x_train, y_train, verbose=1,epochs=100,validation_split=0.2, callbacks=[es,lr,cp]) # epochs default 1
# 둘다 쓰면 search에 넣은걸로 먹힌다
acc = search.score(x_test, y_test)
print("최종 스코어 :", acc)
print(search.best_params_) # 내가 선택한 파라미터들
# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해
print(search.best_score_)

# 최종 스코어 : 0.9652000069618225
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
# 0.9578833182652792

# 최종 스코어 : 0.9810000061988831
# {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'node2': 256, 'node1': 256, 'lr': 0.001, 'drop': 0.1, 'batch_size': 
# 64, 'activation': 'tanh'}
# 0.9758999943733215

