# -*- coding: utf-8 -*-
import numpy as np


def blind_noise_rand():
    batch_size = 128
    image_size = 40

    seed = np.random.randint(5, 30, size=batch_size)
    masks = np.reshape(np.repeat(seed, image_size**2), newshape=[batch_size, image_size, image_size])
    masks = np.expand_dims(masks, axis=3)
    base_n = np.random.normal(0, 1, size=[batch_size, image_size, image_size, 1])
    noise = masks * base_n
    print(noise.shape)
    img = np.ones(shape=[batch_size, image_size, image_size, 1], dtype=np.float32)
    img_deg = img + noise


if __name__ == '__main__':
    blind_noise_rand()

