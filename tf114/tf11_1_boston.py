from sklearn.datasets import load_boston
import tensorflow as tf
tf.compat.v1.set_random_seed(66)


dataset = load_boston()

x_data = dataset.data
y_data = dataset.target

y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) # (442, 13) (442,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
y_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([13,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) # optimizer + train
from sklearn.metrics import r2_score

# r2 = r2_score(y_true, y_pred)

# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
import numpy as np
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(40001):
        cost_val, w_val, b_val, hy_val, _ = sess.run([cost,w,b,hypothesis,train], feed_dict={x:x_train,y:y_train})
        if step %20 == 0:
            print(step, "loss :",cost_val) # epoch, loss
    y_predict = sess.run([hypothesis], feed_dict={x:x_test,y:y_test})
    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(-1,1)
    print(r2_score(y_test, y_predict))
    # 0.8111393181366338

