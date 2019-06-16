#!/bin/sh

python build_masks.py > masks.csv
openssl md5 masks.npy > masks.md5
