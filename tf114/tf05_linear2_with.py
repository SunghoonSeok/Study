import tensorflow as tf
tf.set_random_seed(42) # 랜덤값 고정을 위해 사용

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable([tf.random_normal([1])], name='weight') # [1]은 랜덤한 값 한개를 고르라는 뜻
b = tf.Variable(tf.random_normal([1]), name='bias')


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b)) # [0.8021455] [0.24373285]

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss='mse'

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # optimizer= 'gd'
# # optimizer = tf.train.AdamOptimizer(learning_rate=0.001) # optimizer = 'adam'
# train = optimizer.minimize(cost) # 최소의 loss

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(15001):
#     sess.run(train)
#     if step %20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b)) # epoch, loss, weight, bias
# # 4000 1.1254997e-10 [2.000013] [0.99997246] -> GD lr = 0.01
# # 4000 5.684342e-12 [1.9999971] [1.0000061] -> Adam lr = 0.01
# # 15000 0.0 [1.9999999] [1.0000002] -> Adam lr = 0.001
# # Adam이 gd보다 빠르고 정확함

# sess.close() # 닫아줘야 메모리가 닫힘

# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(15001):
        sess.run(train)
        if step %20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b)) # epoch, loss, weight, bias