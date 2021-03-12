import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

x_train = x_train.reshape(60000, 784)/255.
x_test = x_test.reshape(10000, 784)/255.


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
# x_test = scaler.transform(x_test)

# from sklearn.preprocessing import OneHotEncoder

# y_train = y_train.reshape(-1,1)
# y_val = y_val.reshape(-1,1)
# # y_test = y_test.reshape(-1,1)

# ohencoder = OneHotEncoder()
# ohencoder.fit(y_train)
# y_train = ohencoder.transform(y_train).toarray()
# y_val = ohencoder.transform(y_val).toarray()
# # y_test = ohencoder.transform(y_test).toarray()

y_train = to_categorical(y_train)

x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', [None, 10])

# w = tf.compat.v1.Variable(tf.random.normal([784,256]),name='weight')
# b = tf.compat.v1.Variable(tf.random.normal([1,256]),name='bias')

w1 = tf.compat.v1.Variable(tf.random.normal([784,256],stddev=0.1, name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([256],stddev=0.1, name='bias1'))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# model.add(Dense(10, input_dim=2, activation='relu'))

w2 = tf.compat.v1.Variable(tf.random.normal([256,128],stddev=0.1, name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([128],stddev=0.1, name='bias2'))
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
# model.add(Dense(7, activation='relu'))

w3 = tf.compat.v1.Variable(tf.random.normal([128,64],stddev=0.1, name='weight3'))
b3 = tf.compat.v1.Variable(tf.random.normal([64],stddev=0.1, name='bias3'))
l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([64,10],stddev=0.1, name='weight4'))
b4 = tf.compat.v1.Variable(tf.random.normal([10],stddev=0.1, name='bias4'))
hypothesis = tf.nn.softmax(tf.matmul(l3, w4) + b4)
# model.add(Dense(1, activation='sigmoid))


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy


from sklearn.metrics import accuracy_score
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(301):
        _, cost_val = sess.run([optimizer,loss],feed_dict={x:x_train,y:y_train})
        if step %20 ==0:
            y_pred = sess.run(hypothesis, feed_dict={x:x_test})
            y_pred = np.argmax(y_pred, axis=1)
            print("Epoch :",step,"Loss :",cost_val,"Acc :",accuracy_score(y_test,y_pred))
    # acc = sess.run(accuracy, feed_dict={x:x_test,y:y_test})
    # print("Accuracy :",acc)
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy :",accuracy_score(y_test,y_pred))

# Epoch : 300 Loss : 0.007444849 Acc : 0.9763
# Accuracy : 0.9763

# 2000 0.3249225
# Accuracy : 0.9