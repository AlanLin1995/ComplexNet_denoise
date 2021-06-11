# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import img_2_tfrecord
from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, add

ECNN_model_path = './ECNN_Model/'
ICNN_model_path = './ICNN_Model/'
Comb_model_path = './Comb_Model/'

batch_size = 1
image_size = 224
learn_rate = 0.001
step = 7643*40

def ECNN_residual_block(input_tensor):
    x = Conv2D(64, (3, 3),  activation=None, strides=(1, 1), padding='same')(input_tensor)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (1, 1), activation=None, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)

    y = Activation('relu')(input_tensor)

    x = add([x, y, input_tensor])

    return x


def ECNN_inference(img):

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same')(img)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    for i in range(13):
        x = ECNN_residual_block(x)

    x = Conv2DTranspose(64, (4, 4), activation=None, strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    out = Conv2D(1, (1, 1), activation=None, strides=(1, 1), padding='valid')(x)

    return out


def ICNN_residual_block(input_tensor):
    x = Conv2D(64, (3, 3),  activation=None, strides=(1, 1), padding='same',
               bias_initializer='glorot_normal')(input_tensor)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (1, 1), activation=None, strides=(1, 1), padding='same', bias_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)

    y = Activation('relu')(input_tensor)

    x = add([x, y, input_tensor])

    return x


def ICNN_inference(img):

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same', bias_initializer='glorot_normal')(img)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same', bias_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(2, 2), padding='same', bias_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    for i in range(13):
        x = ICNN_residual_block(x)

    x = Conv2DTranspose(64, (4, 4), activation=None, strides=(2, 2), padding='same',
                        bias_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), activation=None, strides=(1, 1), padding='same', bias_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=-1, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    out = Conv2D(3, (1, 1), activation=None, strides=(1, 1), padding='valid', bias_initializer='glorot_normal')(x)

    return out

def recovery(outting_img, img_cnn_input):
    img_cnn_input_raw = img_cnn_input
    (r_i_B, r_i_G, r_i_R, r_i_e) = cv2.split(img_cnn_input_raw)

    reflection = cv2.merge([r_i_B, r_i_G, r_i_R])
    cv2.imshow('reflection', reflection)

    (i_B, i_G, i_R, i_E) = cv2.split(img_cnn_input)
    (o_B, o_G, o_R) = cv2.split(outting_img)

    c_b = np.sum(np.multiply(i_B, o_B)) / np.sum(np.multiply(o_B, o_B))
    c_g = np.sum(np.multiply(i_G, o_G)) / np.sum(np.multiply(o_G, o_G))
    c_r = np.sum(np.multiply(i_R, o_R)) / np.sum(np.multiply(o_R, o_R))

    o_B = c_b * o_B
    o_G = c_g * o_G
    o_R = c_r * o_R

    outting_img = cv2.merge([o_B, o_G, o_R])

    outting_img[outting_img < 0] = 0
    outting_img[outting_img > 255] = 255
    outting_img = outting_img.astype(np.uint8)

    return outting_img


def midst(edge_img, input_img):
    input_img_3c = tf.slice(input_img, [0, 0, 0, 0], [batch_size, 224, 224, 3])
    input_img_3c = input_img_3c +155 - 115
    edge_img = edge_img -115
    img_cnn_imput = tf.concat([input_img_3c, edge_img], axis=-1)
    return img_cnn_imput

def ECNN_lossing(target_edge, out_edge):
    square = tf.reduce_sum(tf.square(target_edge - out_edge), axis=[1, 2, 3])
    lossing = tf.reduce_mean(square)
    return lossing

def weight_variable():
    x_kernal = tf.constant([[1, 0, -1]], dtype=tf.float32, shape=[1, 3, 1, 1])
    y_kernal = tf.constant([[1], [0], [-1]], dtype=tf.float32, shape=[3, 1, 1, 1])
    return x_kernal, y_kernal

def conv2d(x_kernal, y_kernal, img):
    c_1 = tf.slice(img, [0, 0, 0, 0], [batch_size, 224, 224, 1])
    c_2 = tf.slice(img, [0, 0, 0, 1], [batch_size, 224, 224, 1])
    c_3 = tf.slice(img, [0, 0, 0, 2], [batch_size, 224, 224, 1])
    x_grandt_1 = tf.nn.conv2d(c_1, x_kernal, strides=[1, 1, 1, 1], padding='SAME')
    x_grandt_2 = tf.nn.conv2d(c_2, x_kernal, strides=[1, 1, 1, 1], padding='SAME')
    x_grandt_3 = tf.nn.conv2d(c_3, x_kernal, strides=[1, 1, 1, 1], padding='SAME')
    y_grandt_1 = tf.nn.conv2d(c_1, y_kernal, strides=[1, 1, 1, 1], padding='SAME')
    y_grandt_2 = tf.nn.conv2d(c_2, y_kernal, strides=[1, 1, 1, 1], padding='SAME')
    y_grandt_3 = tf.nn.conv2d(c_3, y_kernal, strides=[1, 1, 1, 1], padding='SAME')
    return x_grandt_1, x_grandt_2, x_grandt_3, y_grandt_1, y_grandt_2, y_grandt_3


def ICNN_lossing(target_img, out_img):
    square = tf.reduce_sum(tf.square(target_img - out_img), axis=[1, 2, 3])
    square = 0.2 * square

    x_kernal, y_kernal = weight_variable()
    t_img_x_g_1, t_img_x_g_2, t_img_x_g_3, t_img_y_g_1, t_img_y_g_2, t_img_y_g_3 = conv2d(x_kernal, y_kernal, target_img)
    o_img_x_g_1, o_img_x_g_2, o_img_x_g_3, o_img_y_g_1, o_img_y_g_2, o_img_y_g_3 = conv2d(x_kernal, y_kernal, out_img)
    bas_char = tf.abs(t_img_x_g_1-o_img_x_g_1) + tf.abs(t_img_x_g_2-o_img_x_g_2)+tf.abs(t_img_x_g_3-o_img_x_g_3) + \
               tf.abs(t_img_y_g_1 - o_img_y_g_1) + tf.abs(t_img_y_g_2 - o_img_y_g_2) + tf.abs(t_img_y_g_3 - o_img_y_g_3)

    l1 = 0.4 * tf.reduce_sum(bas_char, axis=[1, 2, 3])
    lossing = tf.reduce_mean(square + l1)
    return lossing


def lossing(target_edge, out_edge, target_img, out_img):
    ecnn_lossing = ECNN_lossing(target_edge, out_edge)
    icnn_lossing = ICNN_lossing(target_img, out_img)
    lossing = icnn_lossing + 0.4*ecnn_lossing
    return lossing

def optimization(lossing):
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(lossing)
    return train_step

def eval():
    with tf.Graph().as_default():
        input_img = tf.placeholder(shape=[1, 224, 224, 4], dtype=tf.float32)

        with tf.variable_scope('ECNN'):
            ECNN_out_edge = ECNN_inference(input_img)
            var_list = [v for v in tf.all_variables() if v.name.startswith('ECNN')]
            ECNN_saver = tf.train.Saver(var_list)

        K.reset_uids()

        img_cnn_imput = midst(ECNN_out_edge, input_img)

        with tf.name_scope('ICNN'):
            ICNN_out_img = ICNN_inference(img_cnn_imput)
            var_list = [v for v in tf.all_variables() if v.name.startswith('ICNN')]
            ICNN_saver = tf.train.Saver(var_list)


        with tf.Session() as sess:
            ECNN_saver.restore(sess, ECNN_model_path + "ECNN_model_2.ckpt")
            ICNN_saver.restore(sess, ICNN_model_path + "ICNN_model_2.ckpt")

            _, edge_cnn_input, img_cnn_input, ___ = img_2_tfrecord.generateReflectAndEdgeImage('test1.png',
                                                                                               'test1_B.png', tag=1)

            test_img = edge_cnn_input.astype(np.float32)
            test_img = test_img - 155
            # test_img = img_cnn_input.astype(np.float32)
            # test_img = test_img - 115
            # print (test_img)
            test_img = np.expand_dims(test_img, axis=0)
            # outting_img = sess.run(ECNN_out_edge, feed_dict={input_img: test_img, K.learning_phase(): 1})
            outting_img = sess.run(ICNN_out_img, feed_dict={input_img: test_img, K.learning_phase(): 1})
            outting_img = np.squeeze(outting_img, axis=0)
            # print (outting_img.shape)
            outting_img = recovery(outting_img, img_cnn_input)

            cv2.imshow('image', outting_img)
            cv2.waitKey(0)


def train():
    with tf.Graph().as_default():
        t_edge, edge_cnn, image_cnn, t_image = img_2_tfrecord.decode_from_tfrecords()

        input_img = tf.placeholder(shape=[None, 224, 224, 4], dtype=tf.float32)
        target_edge = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)
        target_img = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

        with tf.variable_scope('ECNN'):
            ECNN_out_edge = ECNN_inference(input_img)
            var_list = [v for v in tf.all_variables() if v.name.startswith('ECNN')]
            ECNN_saver = tf.train.Saver(var_list)

        K.reset_uids()

        img_cnn_imput = midst(ECNN_out_edge, input_img)

        with tf.name_scope('ICNN'):
            ICNN_out_img = ICNN_inference(img_cnn_imput)
            var_list = [v for v in tf.all_variables() if v.name.startswith('ICNN')]
            ICNN_saver = tf.train.Saver(var_list)

        lose = lossing(target_edge, ECNN_out_edge, target_img, ICNN_out_img)
        train_step = optimization(lose)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            ECNN_saver.restore(sess, ECNN_model_path + "ECNN_model_1.ckpt")
            ICNN_saver.restore(sess, ICNN_model_path + "ICNN_model_1.ckpt")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global learn_rate
            for i in range(40000):
                if i % (50000) == 0 and i != 0:
                    learn_rate = learn_rate * 0.1
                    print(learn_rate)

                tar_edge, edge_cnn_input, _, tar_img = sess.run([t_edge, edge_cnn, image_cnn, t_image])
                edge_cnn_input = edge_cnn_input - 155

                sess.run(train_step, feed_dict={input_img: edge_cnn_input, target_edge: tar_edge,
                                                target_img: tar_img, K.learning_phase(): 1})

                if i % 100 == 0:
                    lost = sess.run(lose, feed_dict={input_img: edge_cnn_input, target_edge: tar_edge,
                                                    target_img: tar_img, K.learning_phase(): 1})
                    print('step: %d, lossing=%f' % (i, lost))


            save_path_1 = ECNN_saver.save(sess, ECNN_model_path + "ECNN_model_2.ckpt")
            save_path_2 = ICNN_saver.save(sess, ICNN_model_path + "ICNN_model_2.ckpt")
            coord.request_stop()
            coord.join(threads)




if __name__ == "__main__":
    # train()
    eval()



