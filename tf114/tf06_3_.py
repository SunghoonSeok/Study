# tf06_2.py의 lr을 수정해서 epoch가 2000번보다 적게 만들어라
# 100번

import tensorflow as tf
tf.set_random_seed(42) # 랜덤값 고정을 위해 사용

# x_train = [1,2,3]
# y_train = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable([tf.random_normal([1])], name='weight') # [1]은 랜덤한 값 한개를 고르라는 뜻
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.2707471).minimize(cost) # optimizer + train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost) # optimizer + train



# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(101):
        cost_val, W_val, b_val, _ =sess.run([cost,W,b,train], feed_dict={x_train:[1,2,3],y_train:[3,5,7]})
        if step %5 == 0:
            print(step, cost_val, W_val, b_val) # epoch, loss, weight, bias
    
    print(sess.run(hypothesis, feed_dict={x_train:[4]})) # predict
    print(sess.run(hypothesis, feed_dict={x_train:[5,6]})) # predict
    print(sess.run(hypothesis, feed_dict={x_train:[6,7,8]})) # predict

# 500 7.768601e-13 [[2.000001]] [0.99999785]
# [[9.000002]]
# [[11.000003 13.000004]]
# [[13.000004 15.000005 17.000006]]

# Adam 0.27
# 100 2.4825567e-05 [[2.000211]] [0.99946296]
# [[9.000307]]
# [[11.000518 13.00073 ]]
# [[13.00073 15.00094 17.00115]]