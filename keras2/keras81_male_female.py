import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions, preprocess_input
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
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
    batch_size=2000,
    class_mode='binary',
    subset='training'
)
xy_val = train_datagen.flow_from_directory(
    '../data/image/sex/gender',
    target_size=(150,150),
    batch_size=1000,
    class_mode='binary',
    subset='validation'
)
# Found 1389 images belonging to 2 classes.
# Found 347 images belonging to 2 classes.
print(xy_train[0][0].shape) # (14, 150, 150, 3)
print(xy_train[0][1].shape) # (14,)
# test_generator

np.save('../data/image/brain/npy/keras66_train_x.npy', arr=xy_train[0][0])
np.save('../data/image/brain/npy/keras66_train_y.npy', arr=xy_train[0][1])
np.save('../data/image/brain/npy/keras66_val_x.npy', arr=xy_val[0][0])
np.save('../data/image/brain/npy/keras66_val_y.npy', arr=xy_val[0][1])

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
x_val = np.load('../data/image/brain/npy/keras66_val_x.npy')

y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
y_val = np.load('../data/image/brain/npy/keras66_val_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (14, 150, 150, 3) (14,)
# (14, 150, 150, 3) (14,)
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))
vgg16.trainable=True
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D()) # hidden레이어 이전에 사용. 공간 데이터에 대한 글로벌 평균 풀링 작업
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))   # BatchNormalization이후에 액티베이션 함수 사용
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=1e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 40)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)
filepath = 'c:/data/modelcheckpoint/keras81_1_checkpoint.hdf5'
cp = ModelCheckpoint(filepath, save_best_only=True, monitor = 'val_loss')
history = model.fit(x_train,y_train, epochs=500, validation_split=0.2, callbacks=[es,lr,cp])

loss, acc =model.evaluate(x_val, y_val)
print(loss, acc)


xy_pred = test_datagen.flow_from_directory(
    '../data/image/my_face',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
)

np.save('../data/image/brain/npy/keras66_pred_x.npy', arr=xy_pred[0][0])
np.save('../data/image/brain/npy/keras66_pred_y.npy', arr=xy_pred[0][1])
x_pred = np.load('../data/image/brain/npy/keras66_pred_x.npy')
# x_pred = preprocess_input(x_pred)
result = model.predict(x_pred)

print('남자일 확률은 ', result[0]*100,'% 입니다.')
print('남자일 확률은 ', result[1]*100,'% 입니다.')
print('남자일 확률은 ', result[2]*100,'% 입니다.')
print('남자일 확률은 ', result[3]*100,'% 입니다.')
print('남자일 확률은 ', result[4]*100,'% 입니다.')
print('남자일 확률은 ', result[5]*100,'% 입니다.')
print(np.where(result < 0.5, '여자', '남자'))

# 0.31841978430747986 0.8732718825340271
# Found 6 images belonging to 2 classes.
# 남자일 확률은  [46.380447] % 입니다.
# 남자일 확률은  [0.36541945] % 입니다.
# 남자일 확률은  [94.909035] % 입니다.
# 남자일 확률은  [1.5173736] % 입니다.
# 남자일 확률은  [90.198456] % 입니다.
# 남자일 확률은  [45.35643] % 입니다.
# [['여자']
#  ['여자']
#  ['남자']
#  ['여자']
#  ['남자']
#  ['여자']]