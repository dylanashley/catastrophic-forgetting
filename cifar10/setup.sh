#!/bin/sh

# build test states
python build_masks.py 'cifar10_masks.npy' 'cifar10' '10' > cifar10.csv
openssl md5 cifar10_masks.npy > cifar10_masks.md5
