#!/bin/sh

# build test states
python build_masks.py 'cifar100_masks.npy' 'cifar100' '10' > cifar100.csv
openssl md5 cifar100_masks.npy > cifar100_masks.md5
