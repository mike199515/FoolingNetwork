import numpy as np
import tensorflow as tf

import vgg.vgg19 as vgg19
from caffe_classes import class_names
import cv2

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
def load_image(img_path):
    # load image
    img = cv2.imread(img_path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = cv2.resize(crop_img, (224, 224))
    return resized_img

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

########### PARAMS ############

learning_rate=1e-1
origin_output=265
target_output=get_ind("basketball")
train_y_truth=gen_onehot_truth(target_output)
def should_break(lst,step):
    return lst[target_output]>0.99

########### MAIN ############

img1 = load_image("../dog2.png")
batch = img1.reshape((1, 224, 224, 3))

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [1, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    #with tf.name_scope("content_vgg"):
    vgg.build(images)


    y_truth = tf.placeholder(tf.float32, (None, 1000))
    cross_entropy = -tf.reduce_sum(y_truth * tf.log(vgg.prob))


    #training process

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, var_list=[vgg.div])
    sess.run(tf.initialize_all_variables())

    for i in range(20000):
        train_prob = vgg.prob.eval(session=sess, feed_dict=feed_dict)

        if should_break(train_prob[0], i):
            print("break@{}".format(i))
            print("mean std={}".format(cv2.meanStdDev(vgg.div.eval(sess))[1]))
            break

        if True or i % 100 == 0:
            train_entropy = cross_entropy.eval(session=sess, feed_dict={images:batch, y_truth: [train_y_truth]})
            print("====\nstep {}, target_prob={}, train_entropy={}".format(i, train_prob[0][
                target_output],train_entropy))

        train_step.run(session=sess, feed_dict={images: batch, y_truth: [train_y_truth]})

    mod_img = vgg.div_rgb_scaled.eval(session=sess, feed_dict={images:batch})[0]
    cv2.imwrite("dog2->basketball_0.99.png", mod_img)
    # show_img(mod_img)
