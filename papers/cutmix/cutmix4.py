from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np


import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import PIL
from PIL import ImageDraw
from keras import regularizers
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import imgaug as ia
from imgaug import augmenters as iaa
from keras.applications.mobilenet import preprocess_input


def append_ext(fn):
    return fn+".png"
traindf=pd.read_csv('C:/data/image/cifar10/trainLabels.csv',dtype=str)
testdf=pd.read_csv("C:/data/image/cifar10/sampleSubmission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator1=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
color_mode='rgb',
y_col="label",
subset="training",
batch_size=32,
shuffle=True,
class_mode="categorical",
target_size=(32,32))


train_generator2=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
y_col="label",
color_mode='rgb',
subset="training",
batch_size=32,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

valid_generator=datagen.flow_from_dataframe(
dataframe=traindf,
directory="C:/data/image/cifar10/train/train/",
x_col="id",
y_col="label",
color_mode='rgb',
subset="validation",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(32,32))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="C:/data/image/cifar10/test/test/",
x_col="id",
color_mode='rgb',
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))

from cutmix_keras import CutMixImageDataGenerator  # Import CutMix



train_generator = CutMixImageDataGenerator(
    generator1=train_generator1,
    generator2=train_generator2,
    img_size=32,
    batch_size=32,
)


from keras.layers import BatchNormalization
from keras.applications import MobileNet, ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,LSTM, GlobalAveragePooling2D


mobile = MobileNet(weights='imagenet', include_top=False,input_shape=(32,32,3))
# from tensorflow.keras.models import Sequential


model = Sequential()
model.add(mobile)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.Adam(lr=0.0001, decay=1e-5),loss="categorical_crossentropy",metrics=["accuracy"])


# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
es = EarlyStopping(patience=16)
lr = ReduceLROnPlateau(patience=8,factor=0.5)
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator1.n//train_generator1.batch_size,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=500, callbacks= [es,lr]

)


model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_TEST)


test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


sub = pd.read_csv('C:/data/image/cifar10/sampleSubmission.csv')
sub['label'] = predictions
sub.to_csv("C:/data/image/cifar10/results_cutmix.csv",index=False)




# no cutmix
# loss: 0.0481 - accuracy: 0.9843 - val_loss: 0.7864 - val_accuracy: 0.8235
# score : 82

# cutmix
# loss: 1.4698 - accuracy: 0.5917 - val_loss: 0.6165 - val_accuracy: 0.8264
# score : 