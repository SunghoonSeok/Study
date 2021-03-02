from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


y_train = y_train.reshape(y_train.shape[0],)
y_test = y_test.reshape(y_test.shape[0],)

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()


vgg19 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3), alpha=1.4)
vgg19.trainable = True
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

x_train = x_train.astype('float32')/255.  # 전처리
x_test = x_test.astype('float32')/255.  # 전처리

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=21, mode='min')
file_path = 'c:/data/modelcheckpoint/checkpoint_78_8_Mobilenetv2.hdf5'
mc = ModelCheckpoint(file_path, monitor='val_loss',save_best_only=True,mode='min',verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=7,verbose=1,mode='min')
model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2, callbacks=[es,mc,rl])

loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("Loss : ", loss)
print("acc : ", acc)

model2 = load_model('c:/data/modelcheckpoint/checkpoint_78_8_Mobilenetv2.hdf5')
loss2, acc2 = model2.evaluate(x_test, y_test, batch_size=64)
print("Best Loss : ", loss2)
print("Best acc : ", acc2)

# Loss :  0.9184151291847229
# acc :  0.8345999717712402

# Best Loss :  0.607860803604126
# Best acc :  0.8148999810218811