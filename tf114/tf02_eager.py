# 즉시 실행 모드
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf

print(tf.executing_eagerly()) # False

tf.compat.v1.disable_eager_execution() # tensorflow2에서 tensorflow1 코딩을 가능하게 만듬
print(tf.executing_eagerly()) # False

print(tf.__version__)

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session() # 자료형 출력을 위한 -> tensorflow2에서는 삭제됨
sess = tf.compat.v1.Session() # 1.14부터는 session 이렇게
print(sess.run(hello)) # b'Hello World'

