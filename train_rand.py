# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2, os
import model


learn_rate = 0.01
epochs = 140
batch_size = 128
train_data = './data/img_clean_pats.npy'
model_path = './model/'
noise_start = 30
noise_end = 55 + 1


def blind_noise_rand(img_clean):
    b, w, h, c = img_clean.shape

    seed = np.random.randint(noise_start, noise_end, size=b) / 255
    masks = np.reshape(np.repeat(seed, w*h), newshape=[b, w, h])
    masks = np.expand_dims(masks, axis=3)
    base_n = np.random.normal(0, 1, size=[b, w, h, 1])
    noise = masks * base_n

    img_deg = img_clean + noise
    return img_deg


def train():
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.float32, name='learning_rate')
        training = tf.placeholder(tf.bool, name='is_training')
        img_clean = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        img_noise = tf.placeholder(tf.float32, [None, None, None, 1], name='noise_image')
        img_imag = tf.zeros(shape=tf.shape(img_clean))
        Y = model.inference(img_noise, img_imag, is_training=training)
        loss = model.lossing(Y, img_clean, batch_size)
        opt = model.optimization(loss, lr)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=25)

        with tf.Session() as sess:
            data_total = np.load(train_data)
            data_total = data_total.astype(np.float32) / 255.0
            num_example, row, col, chanel = data_total.shape
            numBatch = num_example // batch_size

            learn_rates = learn_rate * np.ones([epochs])
            learn_rates[70:] = learn_rates[0] / 10.0
            learn_rates[100:] = learn_rates[0] / 100.0

            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                full_path = tf.train.latest_checkpoint(model_path)
                epoch_now = int(os.path.basename(full_path).split('.')[0].split('_')[-1])
                step = epoch_now * num_example
                saver.restore(sess, full_path)

                print("Loading " + os.path.basename(full_path) + " to the model")
            else:
                sess.run(init)
                epoch_now = 0
                step = 0

                print("Initialing the model")

            for epoch in range(epoch_now, epochs):
                np.random.shuffle(data_total)
                for batch_id in range(0, numBatch):
                    batch_images = data_total[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                    batch_degrad = blind_noise_rand(batch_images)
                    _ = sess.run(opt, feed_dict={img_clean: batch_images, img_noise: batch_degrad, lr: learn_rates[epoch],
                                                 training: True})
                    step += 1

                    if batch_id % 100 == 0:
                        lost = sess.run(loss, feed_dict={img_clean: batch_images, img_noise: batch_degrad, lr: learn_rates[epoch],
                                                         training: True})
                        print("step = %d, loss = %f" % (step, lost))
                if epoch % 2 == 0:
                    save_path = saver.save(sess, model_path + "Comp_model_blind_rand_" + "weight_" + str(epoch + 1) + ".ckpt")
                    print("+++++ epoch " + str(epoch + 1) + " is saved successfully +++++")
                print("+++++ epoch " + str(epoch + 1) + " is trained successfully +++++")


if __name__ == '__main__':
    train()

    