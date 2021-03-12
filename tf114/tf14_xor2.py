import tensorflow as tf
import numpy as np

tf.compat.v1.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0],[1],[1],[0]])

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2,10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.random.normal([10]), name='bias1')
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# model.add(Dense(10, input_dim=2, activation='relu'))

w2 = tf.compat.v1.Variable(tf.random.normal([10,7]), name='weight2')
b2 = tf.compat.v1.Variable(tf.random.normal([7]), name='bias2')
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
# model.add(Dense(7, activation='relu'))

w3 = tf.compat.v1.Variable(tf.random.normal([7,1]), name='weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(l2, w3) + b3)
# model.add(Dense(1, activation='sigmoid))


cost = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis)) # sigmoid
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ =sess.run([cost,train], feed_dict={x:x_data,y:y_data})
        if step %100 == 0:
            print(step, "cost :",cost_val) # epoch, loss
    
    h, c, a =sess.run([hypothesis,predict,accuracy], feed_dict={x:x_data,y:y_data})
    print("예측값 :",h,"\n", "원래값 :",c,"\n정확도 :",a)