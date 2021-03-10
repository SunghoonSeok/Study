import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.compat.v1.global_variables_initializer() # 변수 초기화

sess.run(init)

print(sess.run(x))

# tf.Session, tf.global_variables_initializer() 로 써도 돌아감, 근데 warning 뜸