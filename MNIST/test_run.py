#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

NUM_FOLDS = 10

# parse arguments
parser = argparse.ArgumentParser(
    description='This is the main MNIST experiment.')
parser.add_argument(
    'prefix',
    type=str,
    help='prefix for all output files')
parser.add_argument(
    'digits',
    type=str,
    help='semicolon separated list of digits to train and test on')
parser.add_argument(
    'seed',
    type=int,
    help='seed for numpy random number generator')
parser.add_argument(
    'fold',
    type=int,
    help='fold index for testing')
subparsers = parser.add_subparsers(
    dest='optimizer',
    help='optimizer for training')
sgd_parser = subparsers.add_parser('sgd')
adam_parser = subparsers.add_parser('adam')
rms_parser = subparsers.add_parser('rms')
for subparser in [sgd_parser, adam_parser, rms_parser]:
    subparser.add_argument(
        'epochs',
        type=str,
        help='semicolon separated list number of epochs to train')
    subparser.add_argument(
        'lr',
        type=str,
        help='learning rate for training')
sgd_parser.add_argument(
    'momentum',
    type=str,
    help='momentum hyperparameter for sgd')
adam_parser.add_argument(
    'beta_1',
    type=str,
    help='beta 1 hyperparameter for adam')
adam_parser.add_argument(
    'beta_2',
    type=str,
    help='beta 2 hyperparameter for adam')
rms_parser.add_argument(
    'rho',
    type=str,
    help='rho hyperparameter for RMSprop')
args = vars(parser.parse_args())
if 'momentum' not in args:
    args['momentum'] = None
if 'beta_1' not in args:
    assert('beta_2' not in args)
    args['beta_1'] = None
    args['beta_2'] = None
if 'rho' not in args:
    args['rho'] = None
for key, value in args.items():
    try:
        args[key] = value.strip('\'')
    except AttributeError:
        pass

if os.path.isfile('./{}results.csv'.format(args['prefix'])):
    sys.stderr.write('WARNING: skipping as {}results.csv already exists\n'.format(args['prefix']))
    sys.exit(0)

import numpy as np
import tensorflow as tf
import warnings

tf.logging.set_verbosity('FATAL')
warnings.filterwarnings('ignore')

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
digits = [int(i) for i in args['digits'].split(';')]
train_mask, test_mask = build_masks(digits, args['fold'])

# build train and test dataset
x_train = raw_x_train[train_mask, ...]
y_train = raw_y_train[train_mask] - 1
x_test = raw_x_train[test_mask, ...]
y_test = raw_y_train[test_mask] - 1

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
    'epochs,'
    'accuracies,'
    'final_accuracy,'
    'digit_predictions',
    file=outfile)
print_prefix = '{},{},{},{},{},{},{},{}'.format(
    args['seed'],
    args['fold'],
    args['optimizer'],
    args['lr'],
    args['momentum'],
    args['beta_1'],
    args['beta_2'],
    args['rho'])

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
def predictions_matrix(model):
    test_predictions = np.argmax(model.predict(x_test), axis=1)
    rv = list()
    for i in digits:
        for j in digits:
            rv.append(np.sum(np.logical_and(test_predictions + 1 == i, y_test + 1 == j)))
    return rv
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
epochs = [int(i) for i in args['epochs'].split(';')]
accuracies = list()
predictions = list()
for epoch in range(1, max(epochs) + 1):
    model.fit(x_train, y_train)
    predictions.append('|'.join([str(i) for i in predictions_matrix(model)]))
    accuracies.append('{0:.6f}'.format(model.evaluate(x_test, y_test)[1]))
    if epoch in epochs:
        print('{},{},{},{},{}'.format(
            print_prefix,
            epoch,
            ';'.join(accuracies),
            accuracies[-1],
            ';'.join(predictions)),
            file=outfile)
model.save('{}model.h5'.format(args['prefix']))

# close results file
outfile.close()
