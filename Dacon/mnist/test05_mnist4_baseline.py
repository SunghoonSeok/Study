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


lr = ReduceLROnPlateau(patience=30,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=90, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    filepath = f'c:/data/test/mnist/checkpoint/mnist_checkpoint4-{nth}.hdf5'
    mc = ModelCheckpoint(filepath, save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    

    
    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid,batch_size=8)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    # epsilon : 0으로 나눠지는 것을 피하기 위함
    learning_history = model.fit_generator(train_generator,epochs=1000, validation_data=valid_generator, callbacks=[es,mc,lr])
    
    # predict
    model.load_weights(filepath)
    result += model.predict_generator(test_generator,verbose=True)/40
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')


sub['digit'] = result.argmax(1)

sub.to_csv('c:/data/test/mnist/submission_mnist4.csv',index=False)
