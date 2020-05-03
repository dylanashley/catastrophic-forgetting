#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import tensorflow as tf

# load mnist
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# select pictures randomly
N = 25
x = [list() for _ in range(N)]
index = np.random.choice(x_test.shape[0], N * N, replace=False)
for i, j in enumerate(index):
    x[i % N].append(x_test[j, :, :])
x = np.array(x)

# merge pictures
z = np.zeros((28 * N, 28 * N), dtype=np.uint8)
for i in range(N):
    for j in range(N):
        x_idx = i * 28
        y_idx = j * 28
        z[x_idx:x_idx + 28, y_idx:y_idx + 28] += x[i, j, :, :]

# save image
img = Image.fromarray(255 - z)
img.save('mnist_sample.png')
