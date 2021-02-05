import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from pandas import read_csv
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
sub = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)

#distribution of label('digit') 
train['digit'].value_counts() # 각 숫자별 몇개인지

# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

plt.imshow(train2[100].reshape(28,28))

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
test2 = test2/255.0

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# # show augmented image data
# sample_data = train2[100].copy()
# sample = expand_dims(sample_data,0)
# sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
# sample_generator = sample_datagen.flow(sample, batch_size=1)

# plt.figure(figsize=(16,10))

# for i in range(9) : 
#     plt.subplot(3,3,i+1)
#     sample_batch = sample_generator.next()
#     sample_image=sample_batch[0]
#     plt.imshow(sample_image.reshape(28,28))

# cross validation
skf = StratifiedKFold(n_splits=40, random_state=66, shuffle=True)


lr = ReduceLROnPlateau(patience=40,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=120, verbose=1)

val_loss_min1 = []
val_loss_min2 = []
val_loss_min3 = []

result1 = 0
result2 = 0
result3 = 0

nth = 0
def convmodel(self):
    
    self.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    self.add(BatchNormalization())
    
    self.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    self.add(BatchNormalization())
    self.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    self.add(BatchNormalization())
    self.add(Conv2D(64,(5,5),activation='relu',padding='same'))
    self.add(BatchNormalization())
    self.add(Conv2D(128,(5,5),activation='relu',padding='same'))
    self.add(BatchNormalization())
    self.add(MaxPooling2D((3,3)))
    
    self.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    self.add(BatchNormalization())
    self.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    self.add(BatchNormalization())
    self.add(MaxPooling2D((3,3)))
    
    self.add(Flatten())

    self.add(Dense(128,activation='relu'))
    self.add(BatchNormalization())
    self.add(Dense(64,activation='relu'))
    self.add(BatchNormalization())

    self.add(Dense(10,activation='softmax'))



for train_index, valid_index in skf.split(train2,train['digit']) :
    mc1 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_1.hdf5', save_best_only=True, verbose=1)
    mc2 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_2.hdf5', save_best_only=True, verbose=1)
    mc3 = ModelCheckpoint(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_3.hdf5', save_best_only=True, verbose=1)

    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    
    train_generator1 = idg.flow(x_train,y_train,batch_size=16, seed=516)
    train_generator2 = idg.flow(x_train,y_train,batch_size=16, seed=1993)
    train_generator3 = idg.flow(x_train,y_train,batch_size=16, seed=821)

    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    convmodel(model1)
    convmodel(model2)
    convmodel(model3)
    
    model1.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    model2.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    model3.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])

    # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_history_1 = model1.fit_generator(train_generator1,epochs=1000, validation_data=valid_generator, callbacks=[es,mc1,lr])
    learning_history_2 = model2.fit_generator(train_generator2,epochs=1000, validation_data=valid_generator, callbacks=[es,mc2,lr])
    learning_history_3 = model3.fit_generator(train_generator3,epochs=1000, validation_data=valid_generator, callbacks=[es,mc3,lr])

    
    # predict
    model1.load_weights(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_1.hdf5')
    model2.load_weights(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_2.hdf5')
    model3.load_weights(f'c:/data/test/mnist/checkpoint/mnist_checkpoint9_3.hdf5')

    result1 += model1.predict_generator(test_generator,verbose=True)/40
    result2 += model2.predict_generator(test_generator,verbose=True)/40
    result3 += model3.predict_generator(test_generator,verbose=True)/40

    
    # save val_loss
    hist1 = pd.DataFrame(learning_history_1.history)
    hist2 = pd.DataFrame(learning_history_2.history)
    hist3 = pd.DataFrame(learning_history_3.history)

    val_loss_min1.append(hist1['val_loss'].min())
    val_loss_min2.append(hist2['val_loss'].min())
    val_loss_min3.append(hist3['val_loss'].min())

    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


sub["predict_1"] = result1.argmax(1)
sub["predict_2"] = result2.argmax(1)
sub["predict_3"] = result3.argmax(1)


from collections import Counter

for i in range(len(sub)) :
    predicts = sub.loc[i, ['predict_1','predict_2','predict_3']]
    sub.at[i, "digit"] = Counter(predicts).most_common(n=1)[0][0]


sub = sub[['id', 'digit']]
sub.to_csv('c:/data/test/mnist/submission_mnist7.csv',index=False)
