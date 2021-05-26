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

def append_ext(fn):
    return fn+".png"
traindf=pd.read_csv('C:/data/image/cifar10/trainLabels.csv',dtype=str)
testdf=pd.read_csv("C:/data/image/cifar10/sampleSubmission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


train_generator=datagen.flow_from_dataframe(
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


# train_generator2=datagen.flow_from_dataframe(
# dataframe=traindf,
# directory="C:/data/image/cifar10/train/train/",
# x_col="id",
# y_col="label",
# color_mode='rgb',
# subset="training",
# batch_size=32,
# shuffle=True,
# class_mode="categorical",
# target_size=(32,32))

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

# from cutmix_keras import CutMixImageDataGenerator  # Import CutMix


# seq = iaa.Sequential(
#     [
#         iaa.Affine(rotate=(-15, 15)),
#         iaa.Fliplr(0.5),
#         iaa.GaussianBlur((0, 2.0)),
#         iaa.ElasticTransformation(alpha=(0, 70), sigma=9),
#         iaa.AdditiveGaussianNoise(scale=(0, 0.05), per_channel=True),
#         iaa.ChannelShuffle(p=0.5),
#     ],
#     random_order=False,
# )

# class CutMixImageDataGenerator():
#     def __init__(self, generator1, generator2, img_size, batch_size):
#         self.batch_index = 0
#         self.samples = generator1.samples
#         self.class_indices = generator1.class_indices
#         self.generator1 = generator1
#         self.generator2 = generator2
#         self.img_size = img_size
#         self.batch_size = batch_size

#     def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
#         self.generator1._set_index_array()
#         self.generator2._set_index_array()

#     def reset(self):
#         self.batch_index = 0
#         self.generator1.reset()
#         self.generator2.reset()
#         self.reset_index()

#     def get_steps_per_epoch(self):
#         quotient, remainder = divmod(self.samples, self.batch_size)
#         return (quotient + 1) if remainder else quotient
    
#     def __len__(self):
#         self.get_steps_per_epoch()

#     def __next__(self):
#         if self.batch_index == 0: self.reset()

#         crt_idx = self.batch_index * self.batch_size
#         if self.samples > crt_idx + self.batch_size:
#             self.batch_index += 1
#         else:  # If current index over number of samples
#             self.batch_index = 0

#         reshape_size = self.batch_size
#         last_step_start_idx = (self.get_steps_per_epoch()-1) * self.batch_size
#         if crt_idx == last_step_start_idx:
#             reshape_size = self.samples - last_step_start_idx
            
#         X_1, y_1 = self.generator1.next()
#         X_2, y_2 = self.generator2.next()
        
#         cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
#         cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
#         label_ratio = cut_ratio.reshape(reshape_size, 1)
#         cut_img = X_2

#         X = X_1
#         for i in range(reshape_size):
#             cut_size = int((self.img_size-1) * cut_ratio[i])
#             y1 = random.randint(0, (self.img_size-1) - cut_size)
#             x1 = random.randint(0, (self.img_size-1) - cut_size)
#             y2 = y1 + cut_size
#             x2 = x1 + cut_size
#             cut_arr = cut_img[i][y1:y2, x1:x2]
#             cutmix_img = X_1[i]
#             cutmix_img[y1:y2, x1:x2] = cut_arr
#             X[i] = cutmix_img
            
#         X = seq.augment_images(X)  # Sequential of imgaug
#         y = y_1 * (1 - (label_ratio ** 2)) + y_2 * (label_ratio ** 2)
#         return X, y

#     def __iter__(self):
#         while True:
#             yield next(self)


# train_generator = CutMixImageDataGenerator(
#     generator1=train_generator1,
#     generator2=train_generator2,
#     img_size=32,
#     batch_size=32,
# )


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])


# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10
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


filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("C:/data/image/cifar10/results.csv",index=False)