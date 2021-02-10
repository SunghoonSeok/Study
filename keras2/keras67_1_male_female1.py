import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten


train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=10,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2505
)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow 또는 flow_from_directory

# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
# Found 1389 images belonging to 2 classes.
# Found 347 images belonging to 2 classes.
print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)
print(xy_val[0][0].shape)

model = Sequential()
model.add(Conv2D(32, 2, padding='same', activation='relu', input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,2, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Conv2D(64,3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 30)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras67_1_checkpoint3.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
history = model.fit_generator(xy_train, steps_per_epoch=(xy_train.samples/xy_train.batch_size), epochs=500, validation_data=xy_val, validation_steps=(xy_val.samples/xy_val.batch_size),
callbacks=[es,cp,lr])
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
loss, acc = model.evaluate_generator(xy_val)


print("loss :", loss)
print("acc :", acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("acc : ",acc)
print("val_acc : ", val_acc)

# import matplotlib.pyplot as plt
# epochs = len(acc)
# x_axis = range(0,epochs)

# fig, ax = plt.subplots()
# ax.plot(x_axis, acc, label='train')
# ax.plot(x_axis, val_acc, label='val')
# ax.legend()
# plt.ylabel('acc')
# plt.title('acc')
# # plt.show()


# fig, ax = plt.subplots()
# ax.plot(x_axis, loss, label='train')
# ax.plot(x_axis, val_loss, label='val')
# ax.legend()
# plt.ylabel('loss')
# plt.title('loss')
# plt.show()

# loss : 0.6158890724182129
# acc : 0.6405529975891113

# loss : 0.9310007691383362
# acc : 0.6774193644523621

# loss : 0.7922691702842712
# acc : 0.725806474685669