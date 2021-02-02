import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
x2 = x.reshape(70000, 28*28)
# 실습
# pca를 통해 0.95 이상인거 몇개?
# pca 배운거 다 집어넣고 확인!!

pca = PCA(n_components=674)
x2 = pca.fit_transform(x2)
print(x2.shape)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=0.999999)+1
# print("cumsum >=0.95 :", cumsum>=0.95)
# print("d :", d) # d : 674
x2_train = x2[:60000,:]
x2_test = x2[60000:,:]
x2_train = x2_train/255.  # 전처리
x2_test = x2_test/255.  # 전처리

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(400, activation='relu', input_shape=(674,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.002)
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min')
modelpath= '../data/modelcheckpoint/m32_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
model.fit(x2_train, y_train, batch_size=32, epochs=70, validation_split=0.2, callbacks=[early_stopping, cp, lr])

# 4. 평가, 예측
loss, acc = model.evaluate(x2_test, y_test, batch_size=32)
y_pred = model.predict(x2_test[:-10])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)
print("y_test : ", y_test[:-10])
print("y_pred : ", y_recovery)
print("loss : ", loss)
print("acc : ", acc)

# d =674
# loss :  0.2988618016242981
# acc :  0.9648000001907349

# node값 수정
# loss :  0.2309943288564682
# acc :  0.9677000045776367

# batch 32
# loss :  0.2963866591453552
# acc :  0.9794999957084656