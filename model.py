# -*- coding: utf-8 -*-
import tensorflow as tf


chanel_num = 64


def CompressComplexConv2d(real, imag, k_shape, b_shape, stride, mul, idx):
    r, c, in_c, out_c = k_shape
    stride2 = [stride[0], stride[1]*mul, stride[2]*mul, stride[3]]
    kernel_r_r = tf.get_variable(name ='RowRealConv_'+str(idx), shape=[1, c, in_c, out_c],
                               initializer=tf.contrib.layers.xavier_initializer())
    kernel_r_c = tf.get_variable(name='ColRealConv_' + str(idx), shape=[r, 1, out_c, out_c],
                                 initializer=tf.contrib.layers.xavier_initializer())
    biase_r = tf.get_variable(name='RealBiase_' + str(idx), shape=b_shape,
                               initializer=tf.constant_initializer(0.0))
    kernel_i_r = tf.get_variable(name='RowImagConv_' + str(idx), shape=[1, c, in_c, out_c],
                               initializer=tf.contrib.layers.xavier_initializer())
    kernel_i_c = tf.get_variable(name='ColImagConv_' + str(idx), shape=[r, 1, out_c, out_c],
                               initializer=tf.contrib.layers.xavier_initializer())
    biase_i = tf.get_variable(name='ImagBiase_' + str(idx), shape=b_shape,
                               initializer=tf.constant_initializer(0.0))

    Inr_kr = tf.nn.conv2d(real, kernel_r_r, stride, padding='SAME')
    Inr_kr = tf.nn.conv2d(Inr_kr, kernel_r_c, stride2, padding='SAME') + biase_r

    Ini_ki = tf.nn.conv2d(imag, kernel_i_r, stride, padding='SAME')
    Ini_ki = tf.nn.conv2d(Ini_ki, kernel_i_c, stride2, padding='SAME') + biase_i

    Inr_ki = tf.nn.conv2d(real, kernel_i_r, stride, padding='SAME')
    Inr_ki = tf.nn.conv2d(Inr_ki, kernel_i_c, stride2, padding='SAME') + biase_i

    Ini_kr = tf.nn.conv2d(imag, kernel_r_r, stride, padding='SAME')
    Ini_kr = tf.nn.conv2d(Ini_kr, kernel_r_c, stride2, padding='SAME') + biase_r

    real_out = Inr_kr - Ini_ki
    imag_out = Inr_ki + Ini_kr
    return real_out, imag_out


def ComplexConv2d(real, imag, k_shape, b_shape, stride, idx):
    kernel_r = tf.get_variable(name ='RealConv_'+str(idx), shape=k_shape,
                               initializer=tf.contrib.layers.xavier_initializer())
    biase_r = tf.get_variable(name='RealBiase_' + str(idx), shape=b_shape,
                               initializer=tf.constant_initializer(0.0))
    kernel_i = tf.get_variable(name='ImagConv_' + str(idx), shape=k_shape,
                               initializer=tf.contrib.layers.xavier_initializer())
    biase_i = tf.get_variable(name='ImagBiase_' + str(idx), shape=b_shape,
                               initializer=tf.constant_initializer(0.0))

    Inr_kr = tf.nn.conv2d(real, kernel_r, stride, padding='SAME') + biase_r
    Ini_ki = tf.nn.conv2d(imag, kernel_i, stride, padding='SAME') + biase_i
    Inr_ki = tf.nn.conv2d(real, kernel_i, stride, padding='SAME') + biase_i
    Ini_kr = tf.nn.conv2d(imag, kernel_r, stride, padding='SAME') + biase_r

    real_out = Inr_kr - Ini_ki
    imag_out = Inr_ki + Ini_kr
    return real_out, imag_out


def CompressComplexDeconv2d(real, imag, k_shape, b_shape, outshape, stride, mul, idx):
    r, c, out_c, in_c = k_shape
    stride2 = [stride[0], stride[1] * mul, stride[2] * mul, stride[3]]
    kernel_r_r = tf.get_variable(name='RowRealConv_' + str(idx), shape=[1, c, out_c, in_c],
                                 initializer=tf.contrib.layers.xavier_initializer())
    kernel_r_c = tf.get_variable(name='ColRealConv_' + str(idx), shape=[r, 1, out_c, out_c],
                                 initializer=tf.contrib.layers.xavier_initializer())
    biase_r = tf.get_variable(name='RealBiase_' + str(idx), shape=b_shape,
                              initializer=tf.constant_initializer(0.0))
    kernel_i_r = tf.get_variable(name='RowImagConv_' + str(idx), shape=[1, c, out_c, in_c],
                                 initializer=tf.contrib.layers.xavier_initializer())
    kernel_i_c = tf.get_variable(name='ColImagConv_' + str(idx), shape=[r, 1, out_c, out_c],
                                 initializer=tf.contrib.layers.xavier_initializer())
    biase_i = tf.get_variable(name='ImagBiase_' + str(idx), shape=b_shape,
                              initializer=tf.constant_initializer(0.0))

    Inr_kr = tf.nn.conv2d_transpose(real, kernel_r_r, outshape, stride2, padding='SAME')
    Inr_kr = tf.nn.conv2d(Inr_kr, kernel_r_c, stride, padding='SAME') + biase_r

    Ini_ki = tf.nn.conv2d_transpose(imag, kernel_i_r, outshape, stride2, padding='SAME')
    Ini_ki = tf.nn.conv2d(Ini_ki, kernel_i_c, stride, padding='SAME') + biase_i

    Inr_ki = tf.nn.conv2d_transpose(real, kernel_i_r, outshape, stride2, padding='SAME')
    Inr_ki = tf.nn.conv2d(Inr_ki, kernel_i_c, stride, padding='SAME') + biase_i

    Ini_kr = tf.nn.conv2d_transpose(imag, kernel_r_r, outshape, stride2, padding='SAME')
    Ini_kr = tf.nn.conv2d(Ini_kr, kernel_r_c, stride, padding='SAME') + biase_r

    real_out = Inr_kr - Ini_ki
    imag_out = Inr_ki + Ini_kr
    return real_out, imag_out


def ComplexDeconv2d(real, imag, k_shape, b_shape, outshape, stride, idx):
    kernel_r = tf.get_variable(name='RealConv_' + str(idx), shape=k_shape,
                               initializer=tf.contrib.layers.xavier_initializer())
    biase_r = tf.get_variable(name='RealBiase_' + str(idx), shape=b_shape,
                              initializer=tf.constant_initializer(0.0))
    kernel_i = tf.get_variable(name='ImagConv_' + str(idx), shape=k_shape,
                               initializer=tf.contrib.layers.xavier_initializer())
    biase_i = tf.get_variable(name='ImagBiase_' + str(idx), shape=b_shape,
                              initializer=tf.constant_initializer(0.0))

    Inr_kr = tf.nn.conv2d_transpose(real, kernel_r, outshape, stride, padding='SAME') + biase_r
    Ini_ki = tf.nn.conv2d_transpose(imag, kernel_i, outshape, stride, padding='SAME') + biase_i
    Inr_ki = tf.nn.conv2d_transpose(real, kernel_i, outshape, stride, padding='SAME') + biase_i
    Ini_kr = tf.nn.conv2d_transpose(imag, kernel_r, outshape, stride, padding='SAME') + biase_r

    real_out = Inr_kr - Ini_ki
    imag_out = Inr_ki + Ini_kr
    return real_out, imag_out


def ComplexBN(real, imag, is_training):
    real_bn = tf.layers.batch_normalization(real, training=is_training)
    imag_bn = tf.layers.batch_normalization(imag, training=is_training)
    return real_bn, imag_bn


def ComplexRelu(real, imag):
    real_relu = tf.nn.relu(real)
    imag_relu = tf.nn.relu(imag)
    return real_relu, imag_relu


def ComplexHswish(real, imag):
    h_r = tf.nn.relu6(real+3) / 6.
    h_i = tf.nn.relu6(imag+3) / 6.
    real_hswish = tf.multiply(real, h_r) - tf.multiply(imag, h_i)
    imag_hswish = tf.multiply(real, h_i) + tf.multiply(imag, h_r)
    return real_hswish, imag_hswish


def residual_block(real, imag, is_training, idx):
    x1, x2 = CompressComplexConv2d(real, imag, [3, 3, chanel_num, chanel_num], [chanel_num], [1, 1, 1, 1], 1, str(idx) + '_1')
    x1, x2 = ComplexBN(x1, x2, is_training)
    x1, x2 = ComplexHswish(x1, x2)

    x1, x2 = CompressComplexConv2d(x1, x2, [3, 3, chanel_num, chanel_num], [chanel_num], [1, 1, 1, 1], 1, str(idx) + '_2')
    x1, x2 = ComplexBN(x1, x2, is_training)

    y1 = 1 * real
    y2 = 1 * imag

    x1 = tf.nn.relu(x1 + y1)
    x2 = tf.nn.relu(x2 + y2)
    return x1, x2


def inference(real, imag, is_training):
    with tf.variable_scope('ComplexNet'):
        x1, x2 = CompressComplexConv2d(real, imag, [3, 3, 1, chanel_num], [chanel_num], [1, 1, 1, 1], 1, idx=0)
        x1, x2 = ComplexHswish(x1, x2)

        x1, x2 = CompressComplexConv2d(x1, x2, [3, 3, chanel_num, chanel_num], [chanel_num], [1, 1, 1, 1], 1, idx=1)
        x1, x2 = ComplexBN(x1, x2, is_training)
        x1, x2 = ComplexHswish(x1, x2)

        deconv_shape = tf.shape(x1)

        x1, x2 = CompressComplexConv2d(x1, x2, [3, 3, chanel_num, chanel_num], [chanel_num], [1, 1, 1, 1], 2, idx=2)
        x1, x2 = ComplexBN(x1, x2, is_training)
        x1, x2 = ComplexHswish(x1, x2)

        x1_encode = 1 * x1
        x2_encode = 1 * x2
        for i in range(9):
            x1, x2 = residual_block(x1, x2, is_training, idx=i+3)
        x1_decode = 1 * x1
        x2_decode = 1 * x2

        x1 = tf.concat([x1_decode, x1_encode], 3)
        x2 = tf.concat([x2_decode, x2_encode], 3)

        x1, x2 = CompressComplexDeconv2d(x1, x2, [3, 3, chanel_num, chanel_num*2], [chanel_num], deconv_shape, [1, 1, 1, 1], 2, idx=16)
        x1, x2 = ComplexBN(x1, x2, is_training)
        x1, x2 = ComplexHswish(x1, x2)

        x1, x2 = CompressComplexConv2d(x1, x2, [3, 3, chanel_num, chanel_num], [chanel_num], [1, 1, 1, 1], 1, idx=17)
        x1, x2 = ComplexBN(x1, x2, is_training)
        x1, x2 = ComplexHswish(x1, x2)

        x1, x2 = CompressComplexConv2d(x1, x2, [3, 3, chanel_num, 1], [1], [1, 1, 1, 1], 1, idx=18)
        out = tf.sqrt(tf.square(x1) + tf.square(x2))
    return out


def lossing(Y, GT, batch_size):
    loss = (1.0 / batch_size) * tf.nn.l2_loss(Y - GT)
    return loss


def optimization(loss, lr):
    optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    return train_op

