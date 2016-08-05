# test of loading saved mnist model

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from IPython import embed


def show_img(img):
    cv2.imshow("", img)
    cv2.waitKey(0)

def get_one_hot(number):
    #we know there are 10 values
    ret= [0] * 10
    ret[number]=1
    return ret


def get_test_input():
    mnist = input_data.read_data_sets('Mnist_data', one_hot=True)
    return mnist.test.images[:1, ]


def show_res(res):
    number=np.argmax(res)
    print(res)
    print("max number is {}".format(number))

test_input = get_test_input()
test_image = test_input.reshape([-1, 28, 28, 1])
print(test_image.shape)

def generate_anti(original_inp, expect_output,should_break,cost_factor=10000,learning_rate=1e-4):
    expect_one_hot_output=get_one_hot(expect_output)
    with tf.Session() as sess:
        x = tf.placeholder('float', shape=[None, 784])
        y_ = tf.placeholder('float', shape=[None, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        def weight_variable(shape):
            intial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(intial)

        def bias_variable(shape):
            intial = tf.constant(0.1, shape=shape)
            return tf.Variable(intial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1]
                                  , strides=[1, 2, 2, 1], padding="SAME")

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        # 我们的mnist图像是28*28，最后一个1代表只有一个色度空间
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        div = tf.Variable(tf.constant(0., shape=[28, 28, 1]), trainable=False)
        div_image = tf.minimum(tf.maximum(div, 0.), 1.)
        h_conv1 = tf.nn.relu(conv2d(div_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # layer 2
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        trainer = tf.train.AdamOptimizer(1e-4)
        train_step=trainer.minimize(cross_entropy)
        correct_prediction = tf.to_float(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        saver = tf.train.Saver()
        cps = tf.train.get_checkpoint_state("save")
        saver.restore(sess, cps.model_checkpoint_path)

        #res = sess.run(y_conv, feed_dict={x: test_input, keep_prob: 1})
        #show_res(res)

        #let train to make it 5
        print("new train")

        l2=tf.nn.l2_loss(div_image-x_image)
        #cost=cost_factor*cross_entropy + l2
        cost=cost_factor*cross_entropy
        temp = set(tf.all_variables())
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost,var_list=[div])
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))


        #I honestly don't know how else to initialize ADAM in TensorFlow.


        for i in range(20000):
            train_y_conv = y_conv.eval(feed_dict={
                x: original_inp, keep_prob: 1.0})

            if should_break(train_y_conv[0]):
                print("break@{}".format(i))
                print("mean std={}".format(cv2.meanStdDev(div.eval(sess))[1]))
                break

            if i % 100==0:
                train_prediction = correct_prediction.eval(feed_dict={
                    x: original_inp, y_: [expect_one_hot_output], keep_prob: 1.0})

                train_l2 = l2.eval(feed_dict={
                    x: original_inp, keep_prob: 1.0})
                show_res(train_y_conv)
                print("====\nstep {}, prediction {}, l2 {}".format(i, train_prediction,train_l2))


                    #show_img(test_input.reshape(28, 28, 1) + div.eval())
                #print(test_input[0])
                #print(div.eval()[:,:,0])
            train_step.run(feed_dict={x: original_inp, y_: [expect_one_hot_output], keep_prob: 0.5})
        print("generate complete:")
        mod_img=div_image.eval(feed_dict={
                x: original_inp, keep_prob: 1.0})[0]
        #show_img(mod_img)
        train_y_conv = y_conv.eval(feed_dict={
            x: original_inp, keep_prob: 1.0})
        show_res(train_y_conv)
        return mod_img,div.eval(sess)


if __name__ == "__main__":

    ORIGINAL_INPUT=test_input
    ORIGIN_OUTPUT=7
    EXPECT_OUT=7
    def should_break(lst):
        #return lst[ORIGIN_OUTPUT]<0.01
        return lst[EXPECT_OUT]<0.9
        #return np.argmax(lst)==EXPECT_OUT
    NAME = "{}->{}-maximized-noise".format(ORIGIN_OUTPUT,EXPECT_OUT)
    mod_img,dev_img=generate_anti(ORIGINAL_INPUT,EXPECT_OUT,should_break,cost_factor=10,learning_rate=1e-2)
    flattened_img=mod_img.reshape(1,784)
    cv2.imwrite("{}.png".format(NAME),mod_img*255)
    cv2.imwrite("{}_dev.png".format(NAME), dev_img * 255)
    np.save(open("{}.np".format(NAME),"wb"),flattened_img)
