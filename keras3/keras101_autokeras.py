import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

model = ak.ImageClassifier(
    # overwrite=True,
    max_trials=2
)

model.fit(x_train, y_train, epochs=2, validation_split=0.2)
results = model.evaluate(x_test, y_test)

print(results)

# [0.05206753686070442, 0.9819999933242798]