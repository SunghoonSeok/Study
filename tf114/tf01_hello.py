import tensorflow as tf
print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session() # string 출력을 위한
print(sess.run(hello)) # b'Hello World'