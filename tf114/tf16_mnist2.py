import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28)/.astype('float32')/255
x_test = x_test.reshape(10000, 28*28)/255.

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float',[None, 10])
w = tf.Variable(tf.random.normal([784,10]),name='weight')
b = tf.Variable(tf.random.normal([10]),name='bias')

# 2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
# predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy
from sklearn.metrics import accuracy_score
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(10001):
        _, cost_val = sess.run([optimizer,loss],feed_dict={x:x_train,y:y_train})
        if step %200 ==0:
            print(step, cost_val)
    # acc = sess.run(accuracy, feed_dict={x:x_test,y:y_test})
    # print("Accuracy :",acc)
    y_pred = sess.run(hypothesis, feed_dict = {x:x_test})
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy :",accuracy_score(y_test,y_pred))
