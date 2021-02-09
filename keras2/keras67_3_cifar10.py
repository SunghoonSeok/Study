import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=42)
from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen2 = ImageDataGenerator(rescale=1./255)
xy_train = datagen.flow(x_train, y=y_train, batch_size=16)
xy_val = datagen2.flow(x_val, y=y_val, batch_size=16)
xy_test = datagen2.flow(x_test, y=y_test, batch_size=16)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout



model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same',
                 strides=1, input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(64, 2))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(xy_train,
          steps_per_epoch= (len(xy_train)/16), epochs=50,
          validation_data=(xy_val),
          validation_steps=(len(xy_val)/16))

loss, acc = model.evaluate(xy_test)
print("loss :", loss)
print("acc :", acc)

# loss : 1.3128342628479004
# acc : 0.5242000222206116