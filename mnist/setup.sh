#!/bin/sh

# build test states
python build_masks.py 'mnist_masks.npy' 'mnist' '10' > mnist_masks.csv
openssl md5 mnist_masks.npy > mnist_masks.md5

# build plots
python plot_mnist_sample.py
