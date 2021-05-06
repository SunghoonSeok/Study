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

from tensorflow.keras.models import load_model

model = load_model('c:/data/modelcheckpoint/ak_save.h5')
model.summary()

results = model.evaluate(x_test, y_test)

print(results)
# [0.0677037462592125, 0.9779000282287598]

model2 = load_model('c:/data/modelcheckpoint/ak_save_best.h5')
model2.summary()

results2 = model2.evaluate(x_test, y_test)

print(results2)
# [0.0677037462592125, 0.9779000282287598]