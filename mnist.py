__author__ = 'burness'
import tensorflow.examples.tutorials.mnist.input_data as input_data
from IPython import embed
print("importing data")
mnist = input_data.read_data_sets('Mnist_data', one_hot=True)
print("import data done")

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder('float',shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
def weight_variable(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)

def bias_variable(shape):
    intial = tf.constant(0.1,shape=shape)
    return tf.Variable(intial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1]
                          ,strides=[1,2,2,1],padding="SAME")

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
# 我们的mnist图像是28*28，最后一个1代表只有一个色度空间
x_image = tf.reshape(x,[-1,28,28,1])
div=tf.Variable(tf.constant(0.,shape=[28,28,1]),trainable=False)
div_image=x_image+div
# 卷积操作，然后Activation function，relu类似于sigmoid函数，具体要见文章
h_conv1 = tf.nn.relu(conv2d(div_image,W_conv1)+b_conv1)
# max 池化，池化的意思就是把一部分的值用与这块值相关的表示，从而减少数据量，常用的有max池化，平均值池化
h_pool1 = max_pool_2x2(h_conv1)


# layer 2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver=tf.train.Saver()

sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        saver.save(sess,"save/mnist_model",global_step=i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))