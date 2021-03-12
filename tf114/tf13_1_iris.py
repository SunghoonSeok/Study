import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(66)

from sklearn.datasets import load_iris
dataset = load_iris()
x_data = dataset.data
y_data = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
# y_test = ohencoder.transform(y_test).toarray()


x = tf.compat.v1.placeholder('float', [None, 4])
y = tf.compat.v1.placeholder('float', [None, 3])

w = tf.compat.v1.Variable(tf.random.normal([4,3]),name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1,3]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy
from sklearn.metrics import accuracy_score
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5001):
        _, cost_val = sess.run([optimizer,loss],feed_dict={x:x_train,y:y_train})
        if step %200 ==0:
            print(step, cost_val)
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis=1)
    # acc = sess.run(accuracy, feed_dict={x:x_test,y:y_test})
    print("Accuracy :",accuracy_score(y_test,y_pred))

# 5000 0.055494953
# Accuracy : 1.0