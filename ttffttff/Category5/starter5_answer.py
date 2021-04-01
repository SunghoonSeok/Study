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
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.


import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'c:/data/tf_certificate/sunspots.csv')

    time_step = []
    sunspots = []

    with open('c:/data/tf_certificate/sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(row[0])
    import pandas as pd
    series = np.array(sunspots)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000


    time_train = time[0:split_time]
    x_train = series[0:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000


    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    valid_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    t=0
    q=0
    for ex in train_set:
        t += 1
    print("t :",t)
    for ex in valid_set:
        q += 1
    print("q :",q)
        
    # x_train=[]
    # y_train=[]
    # for ex in train_set:
    #     x_train.append(ex[0])
    #     y_train.append(ex[1])
    # x_val=[]
    # y_val=[]
    # for dx in valid_set:
    #     x_val.append(dx[0])
    #     y_val.append(dx[1])
    # x_train = np.array(x_train)
    # x_val = np.array(x_val)
    # y_train = np.array(y_train)
    # y_val = np.array(y_val)
    # print(x_train.shape,y_train.shape,x_val.shape,y_val.shape) (96,)(96,)(13,)(13,)
    # print(x_val)

    # x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1],x_train.shape[2],x_train.shape[3])
    # x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1],x_val.shape[2],x_val.shape[3])
    # y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2],y_train.shape[3])
    # y_val = y_val.reshape(y_val.shape[0]*y_val.shape[1],y_val.shape[2],y_val.shape[3])


    # print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)
        

    model = tf.keras.models.Sequential([
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
      tf.keras.layers.Conv1D(128,2,padding='same', input_shape=(None,1),activation='relu'),
      tf.keras.layers.Dense(128),
      tf.keras.layers.Dense(64),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(16),
      tf.keras.layers.Dense(1)
    ])
    model.summary()
    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #
    model.compile(loss='mae', optimizer='adam')
    model.fit(train_set, batch_size=32, epochs=100, validation_data=valid_set)
    loss = model.evaluate(valid_set)
    print(loss)


    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c:/study/ttffttff/category5/mymodel.h5")



# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
#def model_forecast(model, series, window_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(32).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast


#window_size = # YOUR CODE HERE
#rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
#rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

#result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

## To get the maximum score, your model must have an MAE OF .12 or less.
## When you Submit and Test your model, the grading infrastructure
## converts the MAE of your model to a score from 0 to 5 as follows:

#test_val = 100 * result
#score = math.ceil(17 - test_val)
#if score > 5:
#    score = 5

#print(score)
