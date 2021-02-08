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
    batches = [40, 50]
    optimizers = ['adam']
    dropout = [0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = GridSearchCV(model2, hyperparameters, cv=3)
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

# 최종 스코어 : 0.967199981212616
# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# 0.95496666431427
# {'mean_fit_time': array([2.09782155, 1.75763083, 1.55575212, 1.53010003]), 
# 'std_fit_time': array([0.45485351, 0.03792982, 0.09036409, 0.0626698 ]), 
# 'mean_score_time': array([0.60952004, 0.61459851, 0.49034874, 0.51158086]), 
# 'std_score_time': array([0.02068277, 0.04056165, 0.00439272, 0.05357591]), 
# 'param_optimizer': masked_array(data=['adam', 'adam', 'adam', 'adam'],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.2, 0.3, 0.2, 0.3],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_batch_size': masked_array(data=[40, 40, 50, 50],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 
#             'params': [{'optimizer': 'adam', 'drop': 0.2, 'batch_size': 40}, 
#             {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 40}, 
#             {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}, 
#             {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 50}], 
#             'split0_test_score': array([0.95630002, 0.95415002, 0.95744997, 0.95735002]), 
#             'split1_test_score': array([0.95165002, 0.9447    , 0.95120001, 0.95324999]), 
#             'split2_test_score': array([0.95649999, 0.95130002, 0.95625001, 0.95060003]), 
#             'mean_test_score': array([0.95481668, 0.95005002, 0.95496666, 0.95373334]), 
#             'std_test_score': array([0.00224065, 0.00395791, 0.00270811, 0.00277679]), 
#             'rank_test_score': array([2, 4, 1, 3])}

# 최종 스코어 : 0.9648000001907349
# {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adam'}
# 0.9577333529790243
# {'mean_fit_time': array([2.09351993, 1.702185  , 1.54260651, 1.52305555]), 
# 'std_fit_time': array([0.46834317, 0.01097603, 0.06338516, 0.06023835]), 
# 'mean_score_time': array([0.58796231, 0.61132582, 0.4810586 , 0.5143942 ]), 
# 'std_score_time': array([0.01608676, 0.05414734, 0.0041936 , 0.05369227]), 
# 'param_batch_size': masked_array(data=[40, 40, 50, 50],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_drop': masked_array(data=[0.2, 0.3, 0.2, 0.3],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_optimizer': masked_array(data=['adam', 'adam', 'adam', 'adam'],
#              mask=[False, False, False, False],
#        fill_value='?',
#             dtype=object), 
#             'params': [{'batch_size': 40, 'drop': 0.2, 'optimizer': 'adam'}, 
#             {'batch_size': 40, 'drop': 0.3, 'optimizer': 'adam'}, 
#             {'batch_size': 50, 'drop': 0.2, 'optimizer': 'adam'}, 
#             {'batch_size': 50, 'drop': 0.3, 'optimizer': 'adam'}], 
#             'split0_test_score': array([0.96214998, 0.94735003, 0.96130002, 0.95625001]), 
#             'split1_test_score': array([0.95249999, 0.95015001, 0.95380002, 0.95179999]), 
#             'split2_test_score': array([0.95615   , 0.95515001, 0.95810002, 0.95085001]), 
#             'mean_test_score': array([0.95693332, 0.95088335, 0.95773335, 0.95296667]), 
#             'std_test_score': array([0.00397834, 0.00322627, 0.00307282, 0.00235384]), 
#             'rank_test_score': array([2, 4, 1, 3])}