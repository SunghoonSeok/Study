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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])/255.
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])/255.
    y_train = y_train.reshape(y_train.shape[0],)
    y_test = y_test.reshape(y_test.shape[0],)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense
    
    model = Sequential()
    model.add(Conv1D(64,3,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(32,3,padding='same',activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min')
    model.fit(x_train, y_train, batch_size=32, epochs=200, validation_split=0.2, callbacks=[es,lr])
    loss, acc = model.evaluate(x_test,y_test, batch_size=32)
    print(loss, acc)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c:/study/ttffttff/category2/mymodel.h5")

# 0.45170438289642334 0.9003000259399414