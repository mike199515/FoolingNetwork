# test of loading saved mnist model

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from IPython import embed


def show_img(img):
    cv2.imshow("", img)
    cv2.waitKey(0)


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
sess=tf.Session()
def build_model():
    x = tf.placeholder('float', shape=[None, 784],name="x")
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
    div_image = tf.minimum(tf.maximum(x_image + div, 0.), 1.)
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

    keep_prob = tf.placeholder("float",name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="y_conv")

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    trainer = tf.train.AdamOptimizer(1e-4)
    train_step=trainer.minimize(cross_entropy)
    correct_prediction = tf.to_float(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver = tf.train.Saver()
    cps = tf.train.get_checkpoint_state("save")
    saver.restore(sess, cps.model_checkpoint_path)
    #summary_writer = tf.train.SummaryWriter('/home/zhang/tensorboard_log', sess.graph)
    return x,keep_prob,y_conv

def test_model(x,keep_prob,y_conv,custom_input):
    res = sess.run(y_conv, feed_dict={x: custom_input, keep_prob: 1})
    show_res(res)
    return


def gaussian_blur(inp):
    inp_img=inp.reshape(28,28,1)
    blur_img=cv2.GaussianBlur(inp_img,(3,3),0,0)
    cv2.imwrite("5-gaussian-blur.png",blur_img * 255)
    return blur_img.reshape(1,784)

def bilateral_filter(inp,i=3):
    inp_img = inp.reshape(28, 28, 1)
    blur_img = cv2.bilateralFilter(inp_img, i, i*2, i/2.)
    cv2.imwrite("bilateral_filter.png",blur_img*255)
    return blur_img.reshape(1, 784)

def median_blur(inp):
    inp_img=inp.reshape(28,28,1)
    cv2.imwrite("7-origin.png", test_input.reshape(28,28,1) * 255)
    blur_img=cv2.medianBlur(inp_img,3)
    cv2.imwrite("5-median-blur.png", blur_img * 255)
    return blur_img.reshape(1,784)

if __name__ == "__main__":
    mdl=build_model()
    orig=np.load(open("7->3-0.99.np","rb"))
    print("====sync====")
    test_model(*mdl,test_input)
    print("====orig====")
    test_model(*mdl,orig)
    print("====median_blur====")
    test_model(*mdl,median_blur(orig))
    print("====gaussion_blur====")
    test_model(*mdl, gaussian_blur(orig))
    print("====bilateral_filter====")
    test_model(*mdl, bilateral_filter(orig))