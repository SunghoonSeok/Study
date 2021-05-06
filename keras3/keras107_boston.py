import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42)
#1. 데이터 / 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = ak.StructuredDataRegressor(loss='mse',metrics=['mae'],max_trials=2, overwrite=True)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience=20, verbose=1)
lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

model.fit(x_train, y_train, epochs=300, validation_split=0.2, callbacks=[es,lr])
results = model.evaluate(x_test, y_test)

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

model2 = model.export_model()
try:
    model2.save('ak_save_boston', save_format='tf')
except:
    model2.save('ak_save_noston.h5')
best_model = model.tuner.get_best_model()
try:
    best_model.save('ak_save_best_boston', save_format='tf')
except:
    best_model.save('ak_save_best_boston.h5')

from tensorflow.keras.models import load_model
model3 = load_model('ak_save_boston', custom_objects=ak.CUSTOM_OBJECTS)
result_boston = model3.evaluate(x_test, y_test)
y_pred2 = model3.predict(x_test)
r22 = r2_score(y_test, y_pred2)

model4 = load_model('ak_save_best_boston', custom_objects=ak.CUSTOM_OBJECTS)
result_best_boston = model4.evaluate(x_test, y_test)
y_pred3 = model4.predict(x_test)
r23 = r2_score(y_test, y_pred3)

print("result :", results, r2)
print("load_result :", result_boston, r22)
print("load_best :", result_best_boston, r23)
# result : [10.71219253540039, 2.086357355117798] 0.8539255328808752
# load_result : [10.71219253540039, 2.086357355117798] 0.8539255328808752
# load_best : [10.71219253540039, 2.086357355117798] 0.8539255328808752