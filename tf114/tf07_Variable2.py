import tensorflow as tf
x = [1,2,3]
W = tf.Variable([0.3],tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W*x + b

# 실습
#1. sess.run()
#2. InteractiveSession
#3. .eval(session=sess)


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print("aaa :",aaa)
sess.close()

# # sess = tf.InteractiveSession()
# sess = tf.compat.v1.InteractiveSession() # 인터엑티브에는 with문이 안먹힘
# sess.run(tf.compat.v1.global_variables_initializer())
# bbb = hypothesis.eval()
# print("bbb :",bbb)
# sess.close()

with tf.compat.v1.InteractiveSession().as_default() as sess: # with쓰려면 as_default() 써주기!
    sess.run(tf.compat.v1.global_variables_initializer())
    bbb = hypothesis.eval()
    print("bbb :",bbb)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc :",ccc)
sess.close()

# aaa : [1.3       1.6       1.9000001]
# bbb : [1.3       1.6       1.9000001]
# ccc : [1.3       1.6       1.9000001]