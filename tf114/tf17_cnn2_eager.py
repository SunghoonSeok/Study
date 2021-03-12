# 즉시 실행 모드
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) # False

tf.compat.v1.disable_eager_execution() # tensorflow2에서 tensorflow1 코딩을 가능하게 만듬
print(tf.executing_eagerly()) # False

print(tf.__version__) # 2.3.1

import numpy as np
tf.compat.v1.set_random_seed(66)

# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000 / 100

x = tf.compat.v1.placeholder(tf.float32,[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32,[None, 10])

# 2. 모델 구성

# L1.
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 32]) # (kernel_size, channel, filter(output))
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # stiride를 (2,2)로 주려면 [1,2,2,1]
L1 = tf.compat.v1.layers.batch_normalization(L1)
# L1 = tf.compat.v1.layers.BatchNormalization(L1)

print(L1) # Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
# Conv2D(filter, kernel_size, input_shape) Summary
# Conv2D(10, (3,2), input_shape=(7,7,1)) -> (kernel_size*channel(color, input_dim) + bias)*filter(output) -> 70
# 다음 레이어로 갈때 (28,28,32)로 간다 -> padding='SAME'이고 output 32가 channel자리로 간다.
# conv2d 안에 bias 포함되어 있음.

L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1,],padding='SAME')
print(L1) # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2.
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
print(L2) # Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.compat.v1.layers.batch_normalization(L2)
# L2 = tf.compat.v1.layers.BatchNormalization(L2)
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1,],padding='SAME')
print(L2) # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# L3.
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME')
print(L3) # Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
L3 = tf.compat.v1.layers.batch_normalization(L3)
# L3 = tf.compat.v1.layers.BatchNormalization(L3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1,],padding='SAME')
print(L3) # Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

# L4.
w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 128, 64])
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME')
L4 = tf.compat.v1.layers.batch_normalization(L4)
# L4 = tf.compat.v1.layers.BatchNormalization(L4)
print(L4) # Tensor("Conv2D_3:0", shape=(?, 4, 4, 64), dtype=float32)
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1,],padding='SAME')
print(L4) # Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1,2*2*64])
print(L_flat) # Tensor("Reshape:0", shape=(?, 256), dtype=float32)

# L5.
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64], initializer=tf.compat.v1.initializers.he_normal())
b5 = tf.compat.v1.Variable(tf.random.normal([64]), name='b1')
L5 = tf.nn.selu(tf.matmul(L_flat, w5) + b5)
L5 = tf.nn.dropout(L5, rate=0.2)
print(L5) # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# L6.
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32], initializer=tf.compat.v1.initializers.he_normal())
b6 = tf.compat.v1.Variable(tf.random.normal([32]), name='b2')
L6 = tf.nn.selu(tf.matmul(L5, w6) + b6)
print(L6) # Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)

# L7.
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10], initializer=tf.compat.v1.initializers.he_normal())
b7 = tf.compat.v1.Variable(tf.random.normal([10]), name='b3')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7) + b7)
print(hypothesis) # Tensor("Softmax:0", shape=(?, 10), dtype=float32)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)

# 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('Epoch :', '%04d'%(epoch + 1), 'cost = {:.9f}'.format(avg_cost))
print('훈련 끗!!!')

prediction = tf.equal(tf.math.argmax(hypothesis,1), tf.math.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Epoch : 0015 cost = 0.024226983
# Acc : 0.9892 -> No BatchNormalization, Adam(0.0001)

# Epoch : 0015 cost = 0.008968799
# Acc : 0.9908 -> BatchNormalization, Adam(0.0002)

# Epoch : 0015 cost = 0.008411237
# Acc : 0.9912 -> BatchNormalization, Adam(0.0003)

# Epoch : 0015 cost = 0.007219008
# Acc : 0.9913 -> BatchNormalization, Adam(0.0004)