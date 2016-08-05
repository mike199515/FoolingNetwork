################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################
from IPython import embed
from numpy import *
#import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
#from scipy.misc import imread
#from scipy.misc import imresize
#import matplotlib.image as mpimg
#from scipy.ndimage import filters
#import urllib.request, urllib.parse, urllib.error
#from numpy import random

import tensorflow as tf
import cv2
from caffe_classes import class_names
################################################################################
#Helper Functions

class NoIndexFoundException(Exception):
    pass
def get_ind(label_string):
    for ind,name in enumerate(class_names):
        if name==label_string:
            return ind
    raise NoIndexFoundException

def gen_onehot_truth(ind,size=len(class_names)):
    ret=[0.]*size
    ret[ind]=1.
    return ret

################################################################################
#Param
cost_factor=100000
learning_rate=1e-1
origin_output=265
target_output=get_ind("basketball")
train_y_truth=gen_onehot_truth(target_output)
def should_break(lst,step):
    return lst[target_output]>0.99

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


################################################################################
#Read Image


def generate(source,rand):

    ################################################################################

    # (self.feed('data')
    #         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    #         .lrn(2, 2e-05, 0.75, name='norm1')
    #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    #         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
    #         .lrn(2, 2e-05, 0.75, name='norm2')
    #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    #         .conv(3, 3, 384, 1, 1, name='conv3')
    #         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
    #         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
    #         .fc(4096, name='fc6')
    #         .fc(4096, name='fc7')
    #         .fc(1000, relu=False, name='fc8')
    #         .softmax(name='prob'))

    print("loading net")
    net_data = load("bvlc_alexnet.npy",encoding="latin1").item()
    print("net loaded");
    def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



    x = tf.placeholder(tf.float32, (None,) + xdim)
    # div = tf.Variable(tf.constant(0., shape=(1,)+xdim), trainable=False)
    div = tf.Variable(rand, trainable=False)
    div_x = tf.minimum(tf.maximum(x+div, 0), 255.)
    x_mean=tf.reduce_mean(div_x,[1,2,3],keep_dims=True)
    x_norm=div_x-x_mean

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x_norm, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


    #prob
    #softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    t = time.time()

    print("new train")

    l2=tf.nn.l2_loss(div)

    y_truth = tf.placeholder(tf.float32, (None,1000))

    cross_entropy = -tf.reduce_sum(y_truth*tf.log(prob))
    cost=cost_factor*cross_entropy
    correct_prediction = tf.equal(tf.argmax(prob,1), tf.argmax(y_truth,1))

    temp = set(tf.all_variables())
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost,var_list=[div])
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    for i in range(20000):
        train_prob = prob.eval(session=sess,feed_dict={x: [source]})

        if should_break(train_prob[0],i):
            print("break@{}".format(i))
            print("mean std={}".format(cv2.meanStdDev(div.eval(sess))[1]))
            break

        if True or i % 100 == 0:
            train_l2 = l2.eval(session=sess,feed_dict={x: [source]})
            train_entropy = cross_entropy.eval(session=sess, feed_dict={x: [source],y_truth: [train_y_truth]})
            train_cost = cost.eval(session=sess, feed_dict={x: [source], y_truth: [train_y_truth]})
            print("====\nstep {}, target_prob={} l2={}, train_entropy={},train_cost={}".format(i, train_prob[0][target_output], train_l2,
                                                                                 train_entropy,train_cost))

        train_step.run(session=sess,feed_dict={x: [source], y_truth: [train_y_truth]})

    mod_img = div_x.eval(session=sess,feed_dict={x: [source]})[0]
    # show_img(mod_img)
    train_prob = prob.eval(session=sess,feed_dict={x: [source]})

    print("generate complete")
    return mod_img

################################################################################

#Output:
test_img=cv2.imread("dog2.png")[:,:,:3].astype(float32)
rand_img=cv2.imread("cat_rand_source.png")[:,:,:3].astype(float32)
zero_img=np.zeros((227,227,3),dtype="float32")
white_img=np.ones((227,227,3),dtype="float32")*255
#gen_img=generate(zero_img,zero_img)
gen_img=generate(test_img,zero_img)
cv2.imwrite("dog2->basketball_0.99.png",gen_img)