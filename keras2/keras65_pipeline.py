# 61번을 파이프라인으로 구성!!

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

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
    # return {"kerasclassifier__batch_size" : batches, "kerasclassifier__optimizer" : optimizers, "kerasclassifier__drop" : dropout}
    return {"clf__batch_size" : batches, "clf__optimizer" : optimizers, "clf__drop" : dropout}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model2 = KerasClassifier(build_fn=build_model, epochs=1, batch_size=32, verbose=1)
# pipeline = make_pipeline(MinMaxScaler(), model2)
pipeline = Pipeline([('scaler', MinMaxScaler()),('clf',model2)])
kfold = KFold(n_splits=3, random_state=42)
search = GridSearchCV(pipeline, hyperparameters, cv=kfold)
search.fit(x_train, y_train)
acc = search.score(x_test, y_test)
print("최종 스코어 :", acc)
print(search.best_params_) # 내가 선택한 파라미터들
# print(search.best_estimator_) # 모든 파라미터들 근데 케라스 파라미터 인식을 못해
print(search.best_score_)

# 최종 스코어 : 0.9652000069618225
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
# 0.9578833182652792

# 최종 스코어 : 0.960099995136261
# {'kerasclassifier__batch_size': 10, 'kerasclassifier__drop': 0.2, 'kerasclassifier__optimizer': 'adam'}
# 0.9521833459536234