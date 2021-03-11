import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[1,2],[2,3],[3,1],
          [4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],
          [1],[1],[1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y*tf.math.log(hypothesis)+(1-y)*tf.math.log(1-hypothesis)) # sigmoid

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ =sess.run([cost,train], feed_dict={x:x_data,y:y_data})
        if step %100 == 0:
            print(step, "cost :",cost_val) # epoch, loss
    
    h, c, a =sess.run([hypothesis,predict,accuracy], feed_dict={x:x_data,y:y_data})
    print("예측값 :",h,"\n", "원래값 :",c,"\n정확도 :",a)

# 10000 cost : 0.000763931
# 예측값 : [[8.5222048e-08]
#  [1.3378501e-03]
#  [1.4093729e-03]
#  [9.9817240e-01]
#  [9.9999714e-01]
#  [9.9999988e-01]]
#  원래값 : [[0.]
#  [0.]
#  [0.]
#  [1.]
#  [1.]
#  [1.]]
# 정확도 : 1.0