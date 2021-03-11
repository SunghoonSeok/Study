import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73,80,75],
          [93,88,93],
          [85,91,90],
          [96,98,100],
          [73,66,70]]
y_data = [[152],[185],[180],[196],[142]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([3,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b


cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost) # optimizer + train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.17413885).minimize(cost) # optimizer + train



# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(5001):
        cost_val, w_val, b_val, hy_val, _ =sess.run([cost,w,b,hypothesis,train], feed_dict={x:x_data,y:y_data})
        if step %20 == 0:
            print(step, cost_val, w_val, b_val) # epoch, loss, weight, bias
            print(step, "cost :", cost_val, "\n", hy_val)

# 5000 21.343487 [[0.63505846] [0.04842844] [0.20685321]] [118.281654]
# 5000 cost : 21.343487 [[144.99265] [183.72864] [183.12878] [203.43628] [148.6154 ]]