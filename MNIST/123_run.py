#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tensorflow as tf
import warnings

tf.logging.set_verbosity('FATAL')
warnings.filterwarnings('ignore')

NUM_FOLDS = 10

# parse arguments
parser = argparse.ArgumentParser(
    description='This is the main MNIST experiment.')
parser.add_argument(
    'prefix',
    type=str,
    help='prefix for all output files')
parser.add_argument(
    'seed',
    type=int,
    help='seed for numpy random number generator')
parser.add_argument(
    'fold',
    type=int,
    help='fold index for testing on both datasets')
subparsers = parser.add_subparsers(
    dest='optimizer',
    help='optimizer for training on both datasets')
sgd_parser = subparsers.add_parser('sgd')
adam_parser = subparsers.add_parser('adam')
rms_parser = subparsers.add_parser('rms')
for subparser in [sgd_parser, adam_parser, rms_parser]:
    subparser.add_argument(
        't1_epochs',
        type=int,
        help='number of epochs for training on first dataset')
    subparser.add_argument(
        't2_epochs',
        type=int,
        help='number of epochs for training on second dataset')
    subparser.add_argument(
        'lr',
        type=str,
        help='learning rate for training on both datasets')
sgd_parser.add_argument(
    'momentum',
    type=str,
    help='momentum hyperparameter for sgd on both datasets')
adam_parser.add_argument(
    'beta_1',
    type=str,
    help='beta 1 hyperparameter for adam on both datasets')
adam_parser.add_argument(
    'beta_2',
    type=str,
    help='beta 2 hyperparameter for adam on both datasets')
rms_parser.add_argument(
    'rho',
    type=str,
    help='rho hyperparameter for RMSprop on both datasets')
args = vars(parser.parse_args())
if 'momentum' not in args:
    args['momentum'] = None
if 'beta_1' not in args:
    assert('beta_2' not in args)
    args['beta_1'] = None
    args['beta_2'] = None
if 'rho' not in args:
    args['rho'] = None

# seed numpy random number generator
np.random.seed(args['seed'])

# load mnist
mnist = tf.keras.datasets.mnist
(raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = mnist.load_data()
raw_x_train, raw_x_test = raw_x_train / 255.0, raw_x_test / 255.0

# get train and test masks
masks = np.load('masks.npy', allow_pickle=True)
assert(masks.shape[0] > NUM_FOLDS)  # make sure we're not touching the holdout
def build_masks(digits, test_fold):
    train_mask = np.zeros(len(raw_y_train), dtype=bool)
    test_mask = np.copy(train_mask)
    for digit in digits:
        for fold in range(NUM_FOLDS):
            if fold == test_fold:
                test_mask += masks[fold][digit]
            else:
                train_mask += masks[fold][digit]
    return train_mask, test_mask
t1_train_mask, t1_test_mask = build_masks([1, 2], args['fold'])
t2_train_mask, t2_test_mask = build_masks([3], args['fold'])

# build train and test dataset
t1_x_train = raw_x_train[t1_train_mask, ...]
t1_y_train = raw_y_train[t1_train_mask] - 1
t1_x_test = raw_x_train[t1_test_mask, ...]
t1_y_test = raw_y_train[t1_test_mask] - 1
t2_x_train = raw_x_train[t2_train_mask, ...]
t2_y_train = raw_y_train[t2_train_mask] - 1
t2_x_test = raw_x_train[t2_test_mask, ...]
t2_y_test = raw_y_train[t2_test_mask] - 1

# build model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(
        input_shape=(28, 28)),
    tf.keras.layers.Dense(
        100,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.glorot_normal(
            seed=np.random.randint(2 ** 16 - 1))),
    tf.keras.layers.Dense(
        3,
        activation=tf.nn.softmax,
        kernel_initializer=tf.keras.initializers.glorot_normal(
            seed=np.random.randint(2 ** 16 - 1)))])

# open results file
outfile = open('{}results.csv'.format(args['prefix']), 'w')
print(
    'seed,'
    'test_fold,'
    'optimizer,'
    'learning_rate,'
    'momentum,'
    'beta_1,'
    'beta_2,'
    'rho,'
    't1_epochs,'
    't2_epochs,'
    'dataset,'
    'stage,'
    'accuracies,'
    'final_accuracy',
    file=outfile)
print_prefix = '{},{},{},{},{},{},{},{},{},{}'.format(
    args['seed'],
    args['fold'],
    args['optimizer'],
    args['lr'],
    args['momentum'],
    args['beta_1'],
    args['beta_2'],
    args['rho'],
    args['t1_epochs'],
    args['t1_epochs'])

# prepare optimizer
if args['optimizer'] == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        lr=float(args['lr']),
        momentum=float(args['momentum']))
elif args['optimizer'] == 'rms':
    optimizer = tf.keras.optimizers.RMSprop(
        lr=float(args['lr']),
        rho=float(args['rho']))
else:
    assert(args['optimizer'] == 'adam')
    optimizer = tf.keras.optimizers.Adam(
        lr=float(args['lr']),
        beta_1=float(args['beta_1']),
        beta_2=float(args['beta_2']))

# train on the first dataset
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
t1_accuracies = list()
t2_accuracies = list()
for epoch in range(args['t1_epochs']):
    model.fit(t1_x_train, t1_y_train)
    t1_accuracies.append('{0:.6f}'.format(model.evaluate(t1_x_test, t1_y_test)[1]))
    t2_accuracies.append('{0:.6f}'.format(model.evaluate(t2_x_test, t2_y_test)[1]))
print('{},1,pre,{},{}'.format(
    print_prefix,
    ';'.join(t1_accuracies),
    t1_accuracies[-1]),
    file=outfile)
print('{},2,pre,{},{}'.format(
    print_prefix,
    ';'.join(t2_accuracies),
    t2_accuracies[-1]),
    file=outfile)
model.save('{}pre.h5'.format(args['prefix']))

# train on the second dataset
t1_accuracies.clear()
t2_accuracies.clear()
for epoch in range(args['t2_epochs']):
    model.fit(t2_x_train, t2_y_train)
    t1_accuracies.append('{0:.6f}'.format(model.evaluate(t1_x_test, t1_y_test)[1]))
    t2_accuracies.append('{0:.6f}'.format(model.evaluate(t2_x_test, t2_y_test)[1]))
print('{},1,post,{},{}'.format(
    print_prefix,
    ';'.join(t1_accuracies),
    t1_accuracies[-1]),
    file=outfile)
print('{},2,post,{},{}'.format(
    print_prefix,
    ';'.join(t2_accuracies),
    t2_accuracies[-1]),
    file=outfile)
model.save('{}post.h5'.format(args['prefix']))

# close results file
outfile.close()
