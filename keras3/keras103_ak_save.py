import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    loss='mse',
    metrics=['acc']
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(patience=6, verbose=1)
lr = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
mc = ModelCheckpoint('c:/data/modelcheckpoint/auto_checkpoint1.hdf5', save_best_only=True, save_weights_only=True,
                     verbose=1)


model.fit(x_train, y_train, epochs=1, validation_split=0.2, callbacks=[es,lr,mc])
results = model.evaluate(x_test, y_test)

print(results)
model2 = model.export_model()
model2.save('c:/data/modelcheckpoint/ak_save.h5')