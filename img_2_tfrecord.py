# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
from scipy import signal


# the number of the training images
train_num = 7643

# the batch_size
batch_size = 2

# tfrecord name
save_path = "./tfRecods"
noiseImage_path = './image_train/noise_img'
clearImageB_path = './image_train/clear_img'
train_tf_name = os.path.join(save_path, "train.tfrecords")
test_tf_name = os.path.join(save_path, "test.tfrecords")

# build save path
if os.path.isdir(save_path):
    pass
else:
    os.mkdir(save_path)


def convolveImage(img, mask):
    return np.abs(signal.convolve2d(img, mask, mode='same'))


def computingEdge(img):
    # mask
    x_module_0 = np.array([[1, -1]])
    x_module_1 = np.array([[-1, 1]])
    y_module_0 = np.array([[1], [-1]])
    y_module_1 = np.array([[-1], [1]])
    # convolve
    img_edge_x0 = convolveImage(img[:, :, 0], x_module_0) + convolveImage(img[:, :, 1], x_module_0) + \
                  convolveImage(img[:, :, 2], x_module_0)
    img_edge_x1 = convolveImage(img[:, :, 0], x_module_1) + convolveImage(img[:, :, 1], x_module_1) + \
                  convolveImage(img[:, :, 2], x_module_1)
    img_edge_y0 = convolveImage(img[:, :, 0], y_module_0) + convolveImage(img[:, :, 1], y_module_0) + \
                  convolveImage(img[:, :, 2], y_module_0)
    img_edge_y1 = convolveImage(img[:, :, 0], y_module_1) + convolveImage(img[:, :, 1], y_module_1) + \
                  convolveImage(img[:, :, 2], y_module_1)

    img_edge = (img_edge_x0 + img_edge_x1 + img_edge_y0 + img_edge_y1) / 4.0

    return img_edge


def generateReflectAndEdgeImage(image_noise, image_clear, tag=0):
    global noiseImage_path
    global clearImageB_path
    if tag != 0:
        noiseImage_path = './'
        clearImageB_path = './'

    # read image
    print (os.path.join(clearImageB_path, image_clear))
    print (os.path.join(noiseImage_path, image_noise))
    noiseImage = cv2.imread(os.path.join(noiseImage_path, image_noise))
    clearImageB = cv2.imread(os.path.join(clearImageB_path, image_clear))


    # computing edge
    target_edge = computingEdge(clearImageB)
    target_edge = target_edge.astype('uint8')
    noise_edge = computingEdge(noiseImage)
    noise_edge = noise_edge.astype('uint8')

    # cv2.imshow('edge', target_edge)
    # cv2.waitKey(0)

    # mege image channel
    (r_B, r_G, r_R) = cv2.split(noiseImage)
    # (rb_B, rb_G, rb_R) = cv2.split(noise_edge)
    edge_cnn_input = cv2.merge([r_B, r_G, r_R, noise_edge])
    image_cnn_input = cv2.merge([r_B, r_G, r_R,target_edge])


    # return
    return target_edge, edge_cnn_input, image_cnn_input, clearImageB


def write_img2record():

    # writer for save image into tfrecord
    writer_train = tf.python_io.TFRecordWriter(train_tf_name)
    writer_test = tf.python_io.TFRecordWriter(test_tf_name)

    # images
    noiseImage_files = os.listdir(noiseImage_path)
    clearImageB_files = os.listdir(clearImageB_path)
    num_images = len(noiseImage_files)


    for m in range(num_images):

        # read reflection and background image
        # print (str(m)+'.png')
        # print (str(m) + '_B.png')
        target_edge, edge_cnn_input, image_cnn_input, target_image = generateReflectAndEdgeImage(str(m+1)+'_noise.png',
                                                                                                 str(m+1) + '.png')

        # image to bytes
        te_raw = target_edge.tobytes()
        ec_raw = edge_cnn_input.tobytes()
        ti_raw = target_image.tobytes()
        ic_raw = image_cnn_input.tobytes()


        example = tf.train.Example(features=tf.train.Features(feature={
            'target_edge': tf.train.Feature(bytes_list=tf.train.BytesList(value=[te_raw])),
            'edge_cnn_input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ec_raw])),
            'target_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ti_raw])),
            'image_cnn_input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ic_raw]))
        }))

        # write
        if m < train_num:
            writer_train.write(example.SerializeToString())
        else:
            writer_test.write(example.SerializeToString())

    writer_train.close()
    writer_test.close()


def _generate_image_batch_noshuffle(t_edge, edge_cnn, image_cnn, t_image, min_queue_examples, batch_size):
    num_preprocess_threads = 2
    target_edge, edge_cnn_input, image_cnn_input, target_image = tf.train.shuffle_batch(
        [t_edge, edge_cnn, image_cnn, t_image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue = 10)

    return target_edge, edge_cnn_input, image_cnn_input, target_image


def decode_from_tfrecords():
    # generate a queue with filenames
    filename_queue = tf.train.string_input_producer([train_tf_name])

    # define a reader to read image data from tfrecod files
    reader = tf.TFRecordReader()

    # return filename and file
    _, serialized_example = reader.read(filename_queue)

    # features saving the image information, including raw_data, height, width and channel
    features = tf.parse_single_example(serialized_example, features={
                                    'target_edge': tf.FixedLenFeature([], tf.string),
                                    'edge_cnn_input': tf.FixedLenFeature([], tf.string),
                                    'target_image': tf.FixedLenFeature([], tf.string),
                                    'image_cnn_input': tf.FixedLenFeature([], tf.string)
                                    })

    # decode image and reshape based height, width and channel
    # target edge
    target_edge = tf.decode_raw(features['target_edge'], tf.uint8)
    target_edge = tf.reshape(target_edge, [256, 256, 1])
    target_edge = tf.cast(target_edge, tf.float32)

    # edge cnn input
    edge_cnn = tf.decode_raw(features['edge_cnn_input'], tf.uint8)
    edge_cnn = tf.reshape(edge_cnn, [256, 256, 4])
    edge_cnn = tf.cast(edge_cnn, tf.float32)

    # image cnn input
    image_cnn = tf.decode_raw(features['image_cnn_input'], tf.uint8)
    image_cnn = tf.reshape(image_cnn, [256, 256, 4])
    image_cnn = tf.cast(image_cnn, tf.float32)

    # target image
    target_image = tf.decode_raw(features['target_image'], tf.uint8)
    target_image = tf.reshape(target_image, [256, 256, 3])
    target_image = tf.cast(target_image, tf.float32)


    return _generate_image_batch_noshuffle(target_edge, edge_cnn, image_cnn, target_image, 60, batch_size)
    # return target_edge


if __name__ == "__main__":
    write_img2record()
    # target_edge, edge_cnn_input, image_cnn_input, target_image = generateReflectAndEdgeImage(str(1) + '_noise.png',
    #                                                                                          str(1) + '.png')
    # cv2.imshow('target_edge', target_edge)
    # cv2.imshow('edge_cnn_input', edge_cnn_input)
    # cv2.imshow('image_cnn_input', image_cnn_input)
    # cv2.imshow('target_image', target_image)
    # cv2.waitKey(0)

    # pass
    # with tf.Graph().as_default():
    #     t_edge, edge_cnn, image_cnn, t_image = decode_from_tfrecords()
    #     # serialized_example = decode_from_tfrecords(train_tf_name)
    #
    #     with tf.Session() as sess:
    #         step = 5
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #         for i in range(step):
    #             target_edge, edge_cnn_input, image_cnn_input, target_image = sess.run([t_edge, edge_cnn, image_cnn, t_image])
    #             print (target_edge.shape)
    #             print(edge_cnn_input.shape)
    #
    #
    #             # se = sess.run(serialized_example)
    #             # print (se.shape)
    #
    #         coord.request_stop()
    #         coord.join(threads)