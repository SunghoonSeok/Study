import tensorflow as tf
import matplotlib.pyplot as plt

x = [1.,2.,3.]
y = [2.,4.,6.]
W = tf.compat.v1.placeholder(tf.float32)

hypothesis = x*W

cost = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
cost_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i*0.1
        curr_cost = sess.run(cost, feed_dict={W:curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("==========================")
print(w_history)
print("==========================")
print(cost_history)

plt.plot(w_history, cost_history)
plt.show()

# hypothesis나 cost도 계산하는거라 initializer를 해줬던건데
# 얘네는 tensorflow의 variable이 아니라 python의 variable이라 initializer 안해줘도 돌아가는듯