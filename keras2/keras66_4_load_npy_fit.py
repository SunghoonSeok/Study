import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten


x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_val = np.load('../data/image/brain/npy/keras66_val_x.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')

y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_val = np.load('../data/image/brain/npy/keras66_val_y.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)


model = Sequential()
model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(128,3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(3))
model.add(Conv2D(32,3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 50)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras62_1_checkpoint_{val_loss:.4f}-{epoch:02d}.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
history = model.fit(x_train,y_train, epochs=500, validation_data=(x_val,y_val),callbacks=[es])

result = model.evaluate(x_test, y_test)
print(result)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = len(acc)
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, acc, label='train')
ax.plot(x_axis, val_acc, label='val')
ax.legend()
plt.ylabel('acc')
plt.title('acc')
# plt.show()


fig, ax = plt.subplots()
ax.plot(x_axis, loss, label='train')
ax.plot(x_axis, val_loss, label='val')
ax.legend()
plt.ylabel('loss')
plt.title('loss')
plt.show()


