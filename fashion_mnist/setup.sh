#!/bin/sh

# build test states
python build_masks.py 'fashion_mnist_masks.npy' 'fashion_mnist' '10' > fashion_mnist_masks.csv
openssl md5 fashion_mnist_masks.npy > fashion_mnist_masks.md5

# build plots
python plot_fashion_mnist_sample.py
