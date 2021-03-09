# 실습
# 맹그러봐!!!

import pandas as pd

df = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')
print(df.corr)
print(df.head())
print(df.index)

print(df.shape) # 4898,12

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape, y.shape)

print(y.unique()) # 7
print(y.value_counts())

found1 = df[df['quality']==3].index
found2 = df[df['quality']==9].index

df = df.drop(found1)
df = df.drop(found2)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# print(y.unique())
# print(y.value_counts())
# print(x.shape, y.shape)

# df = df.drop(['citric acid', 'free sulfur dioxide'], axis=1)
# x = df.iloc[:,:-1]
# y = df.iloc[:,-1]
x = x.values
y = y.values

print(x.shape, y.shape)

# # 상관계수
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style='dark', font_scale=1.2, font='Malgun Gothic') # , palette='pastel'
# sns.color_palette('Paired',6)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()


from sklearn.preprocessing import OneHotEncoder

y = y.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y)
y = ohencoder.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=66, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, random_state=66, stratify=y_train)


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
model.add(Dense(5,activation='softmax'))

# model.summary()
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr = 0.00005)
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

# Loss : 1.2811334133148193
# Acc : 0.5357142686843872

# Load_loss : 1.0826810598373413
# Load_acc : 0.5520408153533936
