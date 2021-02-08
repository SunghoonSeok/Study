# 노드의 개수 parameter

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

#2. 모델
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
def build_model(drop=0.5, optimizer='adam', node1=128, node2=64, kernel_size=2, pool_size=2, lr=0.001):
    optimizer=optimizer(lr=lr)
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(node1, kernel_size, activation='relu',padding='same', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2,kernel_size, activation='relu',padding='same', name='hidden2')(x)
    x = MaxPooling2D(pool_size)(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dense(64, activation='relu', name='hidden4')(x)
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
    kernel_size = [2, 3, 4]
    pool_size = [2, 3, 4]
    lr=[0.01,0.005,0.001]
    
    # activation1 = ['relu', 'linear'] # 따로 layer 만들어서 변수처리 ㄱㄱ
    return {"batch_size" : batches, "optimizer" : optimizer, "drop" : dropout, "node1": node1, 
    "node2": node2, "kernel_size": kernel_size, "pool_size" :pool_size, "lr":lr}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv=3)
search.fit(x_train, y_train, verbose=1)
acc = search.score(x_test, y_test)
print("최종 스코어 :", acc)
print(search.best_params_) # 내가 선택한 파라미터들
# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해
print(search.best_score_)

# 최종 스코어 : 0.9652000069618225
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
# 0.9578833182652792

# 최종 스코어 : 0.9857000112533569
# {'pool_size': 4, 'optimizer': 'adam', 'node2': 256, 'node1': 256, 'kernel_size': 4, 'drop': 0.1, 'batch_size': 10}
# 0.9823833306630453

# 최종 스코어 : 0.9876999855041504
# {'pool_size': 4, 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'node2': 128, 'node1': 128, 'kernel_size': 4, 'drop': 0.1, 'batch_size': 64}
# 0.9826666514078776

# 최종 스코어 : 0.9854000210762024
# {'pool_size': 2, 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'node2': 256, 'node1': 128, 'lr': 0.001, 'kernel_size': 3, 'drop': 0.2, 'batch_size': 64}
# 0.980316678682963