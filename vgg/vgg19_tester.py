import numpy as np
import tensorflow as tf

import vgg.vgg19 as vgg19

import cv2

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

########### MAIN ############

img1 = load_image("./vgg_test.png")
img2 = load_image("./test_data/puzzle.jpeg")
batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    #with tf.name_scope("content_vgg"):
    vgg.build(images)

    sess.run(tf.initialize_all_variables())

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    print(prob)
    print_prob(prob[0], './synset.txt')
    print_prob(prob[1], './synset.txt')
