# 가중치 저장할 것
#1. model.save()
#2. pickle 쓸것

import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()


#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10]
    optimizers = ['adam']
    dropout = [0.2]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = RandomizedSearchCV(model2, hyperparameters, cv=3, refit=True)
search.fit(x_train, y_train, verbose=1)
acc = search.score(x_test, y_test)
print("최종 스코어 :", acc)
print(search.best_params_) # 내가 선택한 파라미터들
# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해
print(search.best_score_)

import pickle
# import joblib
# pickle.dump(search.best_estimator_.,open('../data/h5/keras64_pickle.dat', 'wb')) 
search.best_estimator_.model.save('../data/h5/keras64_pickle.h5')

# model3 = pickle.load(open('../data/h5/keras64_pickle.dat', 'rb'))
# # model2 = joblib.load('../data/xgb_save/m39.joblib.dat')
model3 = load_model('../data/h5/keras64_pickle.h5')
# # model3.load_model('../data/xgb_save/m39.xgb.model')
# print('불러왔다!')
y_pred = model3.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

acc2 = accuracy_score(y_test, y_pred)
print('acc2 :',acc2)

# 최종 스코어 : 0.967199981212616
# {'batch_size': 10, 'drop': 0.2, 'optimizer': 'adam'}
# 0.9536666671435038
# acc2 : 0.9672