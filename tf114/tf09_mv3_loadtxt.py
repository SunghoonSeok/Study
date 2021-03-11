import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(66)

dataset = np.loadtxt('c:/data/csv/data-01-test-score.csv', delimiter=',')

x_data = dataset[:,:-1]
y_data = dataset[:,-1:]
print(x_data.shape,y_data.shape) # (25, 3) (25,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b


cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost) # optimizer + train
# train = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost) # optimizer + train

# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost) # optimizer + train



# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(13001):
        cost_val, w_val, b_val, hy_val, _ =sess.run([cost,w,b,hypothesis,train], feed_dict={x:x_data,y:y_data})
        if step %20 == 0:
            print(step, "cost :",cost_val,"w :", w_val, "b :",b_val) # epoch, loss, weight, bias
            print(step, "cost :", cost_val, "\n", hy_val)
    print(sess.run(hypothesis, feed_dict={x:x_data[:5]}))
# 13000 cost : 5.7378054 w : [[0.35594022][0.5425205 ][1.1674458 ]] b : [-4.3360205]
# [[152.6077]]
# [[185.08069]]
# [[181.78215]]
# [[199.74583]]
# [[139.17519]]