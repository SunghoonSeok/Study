import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.Variable(tf.random.normal([1]), name='weight1')
w2 = tf.Variable(tf.random.normal([1]), name='weight2')
w3 = tf.Variable(tf.random.normal([1]), name='weight3')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.012)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(20001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if step % 10 == 0:
        print(step, "cost :", cost_val, "\n", hy_val)

# 20000 cost : 0.10452889
# [151.7248  184.49684 180.29439 196.2285  142.23422]