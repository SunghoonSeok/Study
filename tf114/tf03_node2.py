# 실습
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 맹그러

import tensorflow as tf

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(3.0)
sess = tf.Session()

# 덧셈
node3 = tf.add(node1, node2)
# node4 = tf.add_n(node1, node2) # 실행안됨
print('sess.run(node3) :', sess.run(node3)) # sess.run(node3) : 5.0
# print('sess.run(node4) :', sess.run(node4))

# 뺄셈
node3 = tf.subtract(node1, node2)
print('sess.run(node3) :', sess.run(node3)) # sess.run(node3) : -1.0

# 곱셈
node3 = tf.multiply(node1, node2)
# node4 = tf.matmul(node1, node2) # 실행안됨
print('sess.run(node3) :', sess.run(node3)) # sess.run(node3) : 6.0
# print('sess.run(node4) :', sess.run(node4))

# 나눗셈
node3 = tf.divide(node1, node2)
print('sess.run(node3) :', sess.run(node3)) # sess.run(node3) : 0.6666667

# 나머지
# node3 = tf.mod(node1, node2) 
# WARNING:tensorflow:From c:\Study\tf114\tf03_node2.py:39: The name tf.mod is deprecated. Please use tf.math.mod instead.
node3 = tf.math.mod(node1, node2)
print('sess.run(node3) :', sess.run(node3)) # sess.run(node3) : 2.0
