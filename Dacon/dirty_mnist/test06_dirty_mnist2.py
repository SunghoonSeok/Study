import numpy as np
import pandas as pd
import PIL
from numpy import asarray
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
import scipy.signal as signal
# dirty 데이터는 train 데이터 훈련시키자!
# 50000개 
# dirty_mnist_2nd_answer.csv 는 dirty의 y값 

# test_dirty 데이터는 test 데이터!
# 5000개 
# y값을 찾는것이 목표

# img=[]
# for i in range(50000):
#     filepath='c:/data/test/dirty_mnist/dirty_mnist_2nd/%05d.png'%i
#     image=Image.open(filepath)
#     image = image.resize((128,128))
#     image_data=asarray(image)
#     image_data = signal.medfilt2d(np.array(image_data), kernel_size=3)
#     img.append(image_data)
# img2=[]
# for i in range(50000, 55000):
#     filepath='c:/data/test/dirty_mnist/test_dirty_mnist_2nd/%05d.png'%i
#     image2=Image.open(filepath)
#     image2 = image2.resize((128,128))
#     image_data2=asarray(image2)
#     image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
#     img2.append(image_data2)

# np.save('c:/data/test/dirty_mnist/temporary/test.npy', arr=img)
# np.save('c:/data/test/dirty_mnist/temporary/test3.npy', arr=img2)

# img_ch_np=np.load('c:/data/test/dirty_mnist/temporary/test.npy')
# img_ch_np2=np.load('c:/data/test/dirty_mnist/temporary/test3.npy')

# img_ch_np = img_ch_np/255.
# # threshold = 0.98
# # img_ch_np[img_ch_np < threshold] = 0

# img_ch_np2 = img_ch_np2/255.
# # threshold = 0.98
# # img_ch_np2[img_ch_np2 < threshold] = 0

# np.save('c:/data/test/dirty_mnist/temporary/test2.npy', arr=img_ch_np)
# np.save('c:/data/test/dirty_mnist/temporary/test4.npy', arr=img_ch_np2)
x_data = np.load('c:/data/test/dirty_mnist/temporary/test2.npy')
x_test = np.load('c:/data/test/dirty_mnist/temporary/test4.npy')

x_data = x_data.reshape(50000, 128, 128, 1)
x_test = x_test.reshape(5000, 128, 128, 1)


dataset = pd.read_csv('c:/data/test/dirty_mnist/dirty_mnist_2nd_answer.csv')
submission = pd.read_csv('c:/data/test/dirty_mnist/sample_submission.csv')
import matplotlib.pyplot as plt
y_data = dataset.iloc[:,:]
print(y_data)

# plt.figure(figsize=(20, 5))
# ax = plt.subplot(2, 10, 1)
# plt.imshow(x_test[0])


# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.show()

from tensorflow.keras.optimizers import Adam
def convmodel():
    model = Sequential()
    model.add(Conv2D(64, 3, padding='same', activation='relu', input_shape=(128,128,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,padding='same',activation='relu'))
    model.add(AveragePooling2D(3))
    model.add(Conv2D(64,2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,2,padding='same',activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
for i in alphabet:
    y_train = y_train.loc[:,i]
    y_val = y_val.loc[:,i] 
    model = convmodel()
    checkpoint = ModelCheckpoint(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', 
    monitor='val_loss', save_best_only=True, verbose=1)
    lr = ReduceLROnPlateau(patience=10,verbose=1,factor=0.5) #learning rate scheduler
    es = EarlyStopping(patience=25, verbose=1)
    model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), batch_size=32, callbacks=[checkpoint,lr,es])
    model2 = load_model(f'c:/data/test/dirty_mnist/checkpoint/checkpoint-{i}.hdf5', compile=False)
    y_pred = model.predict(x_test)
    print(y_pred)
    y_recovery = np.where(y_pred<0.5, 0, 1)
    print(y_recovery)
    submission[i] = y_recovery
submission.to_csv('c:/data/test/dirty_mnist/submission.csv', index=False)
