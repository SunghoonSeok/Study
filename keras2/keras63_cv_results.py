# 61 카피해서
# model.cv_results를 붙여서 완성

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

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
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

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
print(search.cv_results_)

# 최종 스코어 : 0.9652000069618225
# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
# 0.9578833182652792

# 최종 스코어 : 0.9678000211715698
# {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}
# 0.955383320649465
# {'mean_fit_time': array([7.00051936, 3.82475026, 3.13480202, 2.72680179, 1.83340669,
#        5.80273461, 5.55144986, 1.87615339, 1.52712528, 2.62377397]), 'std_fit_time': array([0.49857986, 0.09017939, 0.05285451, 0.10717602, 
# 0.07282085,
#        0.10560144, 0.1591102 , 0.08498088, 0.01560473, 0.01173559]), 'mean_score_time': array([1.99760707, 0.96478677, 1.01802929, 0.85364238, 0.47893985,
#        1.81543167, 1.84809399, 0.47880665, 0.50167656, 0.85119939]), 'std_score_time': array([0.10958189, 0.01251725, 0.0458842 , 0.00544165, 0.01520402,
#        0.01121024, 0.02378735, 0.02227226, 0.06532389, 0.01170892]), 'param_optimizer': masked_array(data=['rmsprop', 'rmsprop', 'adam', 'adam', 'rmsprop',
#                    'adadelta', 'adam', 'rmsprop', 'adadelta', 'adadelta'],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.3, 0.2, 0.3],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[10, 20, 20, 30, 50, 10, 10, 50, 50, 30],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 20}, {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 30}, {'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 10}, {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 10}, {'optimizer': 'rmsprop', 'drop': 0.3, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.2, 'batch_size': 50}, {'optimizer': 'adadelta', 'drop': 0.3, 'batch_size': 30}], 'split0_test_score': array([0.95784998, 0.95389998, 0.95375001, 0.95625001, 0.95300001,
#        0.29574999, 0.95225   , 0.95459998, 0.22930001, 0.22995   ]), 'split1_test_score': array([0.95415002, 0.95200002, 0.94564998, 0.95525002, 0.95494998,
#        0.37255001, 0.94185001, 0.95254999, 0.17075001, 0.18515   ]), 'split2_test_score': array([0.94774997, 0.94615   , 0.95424998, 0.95045   , 0.95819998,
#        0.29745001, 0.95585001, 0.95674998, 0.26719999, 0.2018    ]), 'mean_test_score': array([0.95324999, 0.95068334, 0.95121666, 0.95398335, 0.95538332,
#        0.32191667, 0.94998334, 0.95463332, 0.22241667, 0.20563333]), 'std_test_score': array([0.00417214, 0.00329806, 0.00394152, 0.00253158, 0.00214488,
#        0.0358099 , 0.00593595, 0.0017148 , 0.03967522, 0.01848929]), 'rank_test_score': array([ 4,  6,  5,  3,  1,  8,  7,  2,  9, 10])}