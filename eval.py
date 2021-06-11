# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2, os
import model


sigma = 25
model_path = './model/'
model_name = 'Comp_model_blind_rand_weight_'
test_img = './dataset/dncnn-img/test/BSD68/test001.png'


def psnr(img1, img2):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if(len(img1.shape) == 2):
        m, n = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        m, n, k = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2**B - 1
    MSE = np.sum(diff) / (m * n * k)
    sqrt_MSE = np.sqrt(MSE)
    PSNR = 20 * np.log10(MAX / sqrt_MSE)

    return PSNR


def post_process(img):
    img = np.squeeze(img)
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def eval():
    with tf.Graph().as_default():
        img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        training = tf.placeholder(tf.bool, name='is_training')
        img_noise = img_clean + tf.random_normal(shape=tf.shape(img_clean), stddev=sigma / 255.0)
        img_imag = tf.zeros(shape=tf.shape(img_clean))
        Y = model.inference(img_noise,img_imag, is_training=training)

        var_list = [v for v in tf.all_variables() if v.name.startswith('ComplexNet')]
        saver = tf.train.Saver(var_list)

        with tf.Session() as sess:
            saver.restore(sess, model_path + model_name + ".ckpt")

            img_raw = cv2.imread(test_img, 0)
            img = img_raw.astype(np.float) / 255
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=3)

            for i in range(1):
                out, img_n = sess.run([Y, img_noise], feed_dict={img_clean: img, training: False})
                out = post_process(out)
                img_n = post_process(img_n)

                print('psnr: ', psnr(out, img_raw))
                cv2.imshow('out', out)
                cv2.imshow('img_n', img_n)

                cv2.waitKey(0)


if __name__ == '__main__':
    eval()
    
    