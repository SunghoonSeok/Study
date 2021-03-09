import pandas as pd
wine = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')

y= wine['quality']
x = wine.drop('quality', axis=1)

newlist=[]
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist
import numpy as np
y = np.array(y)
# print(y)
x = x.values



from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohencoder = OneHotEncoder()
ohencoder.fit(y)
y = ohencoder.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, random_state=66)


from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, PowerTransformer
# scaler = QuantileTransformer(n_quantiles=100)
scaler = PowerTransformer()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(11,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

# model.summary()
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=60, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.5, mode='min', verbose=1)
file_path = 'c:/data/modelcheckpoint/checkpoint_86_wine_quality.hdf5'
mc = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=8, epochs=1000, validation_data=(x_val,y_val), callbacks=[es,lr,mc])

loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("Loss :", loss)
print("Acc :", acc)

from sklearn.metrics import r2_score, accuracy_score
y_pred = model.predict(x_test)

model2 = load_model(file_path)
loss2, acc2 = model2.evaluate(x_test, y_test, batch_size=8)
y_pred2 = model2.predict(x_test)

print("Load_loss :", loss2)
print("Load_acc :", acc2)


# Loss : 1.9861623048782349
# Acc : 0.9326530694961548

# Load_loss : 1.1908169984817505
# Load_acc : 0.9346938729286194