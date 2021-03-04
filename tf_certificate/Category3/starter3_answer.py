# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'c:/data/tf_certificate/rps.zip')
    local_zip = 'c:/data/tf_certificate/rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('c:/data/tf_certificate/tmp/')
    zip_ref.close()


    TRAINING_DIR = "c:/data/tf_certificate/tmp/rps/"
    training_datagen = ImageDataGenerator(
    # YOUR CODE HERE
    rescale=1/255.,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=10,
    fill_mode='nearest',
    validation_split=0.2
    )

    train_generator = training_datagen.flow_from_directory(
        'c:/data/tf_certificate/tmp/rps/',
        target_size=(150,150),
        batch_size=12,
        subset='training'
    )# YOUR CODE HERE
    test_generator = training_datagen.flow_from_directory(
        'c:/data/tf_certificate/tmp/rps/',
        target_size=(150,150),
        batch_size=12,
        subset='validation'

    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense


    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(128,3, padding='same',activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,2,padding='same',activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32,2,activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(16,activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-5),metrics=['acc'])
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    es = EarlyStopping(monitor = 'val_loss', patience = 30)
    lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 15, factor = 0.5, verbose = 1)
    model.fit_generator(train_generator, steps_per_epoch=(train_generator.samples/train_generator.batch_size),
                        validation_data=test_generator,validation_steps=(test_generator.samples/test_generator.batch_size),
                        epochs=200,callbacks=[es,lr])
    loss,acc =model.evaluate_generator(test_generator)
    print(loss, acc)
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c:/study/tf_certificate/category3/mymodel.h5")
