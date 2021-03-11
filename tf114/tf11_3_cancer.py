from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.compat.v1.set_random_seed(66)


dataset = load_breast_cancer()

x_data = dataset.data
y_data = dataset.target

y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape) # (569, 30) (569 ,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
y_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([30,1]), name='weight') # 초기값때매 발산하면 tf.zero써
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y)) # loss='mse'

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) # optimizer + train

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32)) # accuracy

# r2 = r2_score(y_true, y_pred)

# with문 사용해서 자동으로 sess가 닫히도록 할수도 있다.
import numpy as np
with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(30001):
        cost_val, _ =sess.run([cost,train], feed_dict={x:x_data,y:y_data})
        if step %100 == 0:
            print(step, "cost :",cost_val) # epoch, loss
    
    h, c, a =sess.run([hypothesis,predict,accuracy], feed_dict={x:x_data,y:y_data})
    print("정확도 :",a)

# 30000 cost : 0.06534101
# 정확도 : 0.9578207