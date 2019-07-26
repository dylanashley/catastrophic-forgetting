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
    'seed',
    type=int,
    help='seed for numpy random number generator')
parser.add_argument(
    'fold',
    type=int,
    help='fold index for testing on both datasets')
parser.add_argument(
    'architecture',
    type=str,
    help='colon separated architecture of the hidden layers of the network')
subparsers = parser.add_subparsers(
    dest='optimizer',
    help='optimizer for training on both datasets')
sgd_parser = subparsers.add_parser('sgd')
adam_parser = subparsers.add_parser('adam')
rms_parser = subparsers.add_parser('rms')
for subparser in [sgd_parser, adam_parser, rms_parser]:
    subparser.add_argument(
        'epochs',
        type=str,
        help='colon separated list number of epochs to train on both datasets')
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
layers = list()
layers.append(tf.keras.layers.Flatten(input_shape=(28, 28)))
for hidden_units in [int(i) for i in args['architecture'].split(':')]:
    layers.append(tf.keras.layers.Dense(
        hidden_units,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=np.random.randint(2 ** 16 - 1))))
layers.append(tf.keras.layers.Dense(
    3,
    activation=tf.nn.softmax,
    kernel_initializer=tf.keras.initializers.RandomNormal(
        seed=np.random.randint(2 ** 16 - 1))))
model = tf.keras.models.Sequential(layers)

# open results file
outfile = open('{}results.csv'.format(args['prefix']), 'w')
print(
    'seed,'
    'test_fold,'
    'architecture,'
    'optimizer,'
    'learning_rate,'
    'momentum,'
    'beta_1,'
    'beta_2,'
    'rho,'
    'epochs,'
    'stage,'
    't1_accuracies,'
    't2_accuracies,'
    't1_final_accuracy,'
    't2_final_accuracy,'
    'predictions',
    file=outfile)
print_prefix = '{},{},{},{},{},{},{},{},{}'.format(
    args['seed'],
    args['fold'],
    args['architecture'],
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

# train on the both datasets
def predictions_matrix(model, x_test, y_test):
    test_predictions = np.argmax(model.predict(x_test), axis=1)
    rv = list()
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            rv.append(np.sum(np.logical_and(test_predictions + 1 == i, y_test + 1 == j)))
    return np.array(rv)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
weights_filename = '{}model_weights.h5'.format(args['prefix'])
epochs = [int(i) for i in args['epochs'].split(':')]
t1_accuracies = list()
t2_accuracies = list()
predictions = list()
for t1_epoch in range(1, max(epochs) + 1):
    model.fit(t1_x_train, t1_y_train)
    t1_accuracies.append('{0:.6f}'.format(model.evaluate(t1_x_test, t1_y_test)[1]))
    t2_accuracies.append('{0:.6f}'.format(model.evaluate(t2_x_test, t2_y_test)[1]))
    predictions.append(':'.join([str(i) for i in \
        predictions_matrix(model, t1_x_test, t1_y_test) + predictions_matrix(model, t2_x_test, t2_y_test)]))
    if t1_epoch in epochs:
        print('{},{},pre,{},{},{},{},{}'.format(
            print_prefix,
            t1_epoch,
            '_'.join(t1_accuracies),
            '_'.join(t2_accuracies),
            t1_accuracies[-1],
            t2_accuracies[-1],
            '_'.join(predictions)),
            file=outfile)
        model.save_weights(weights_filename)
        for t2_epoch in range(1, t1_epoch + 1):
            model.fit(t2_x_train, t2_y_train)
            t1_accuracies.append('{0:.6f}'.format(model.evaluate(t1_x_test, t1_y_test)[1]))
            t2_accuracies.append('{0:.6f}'.format(model.evaluate(t2_x_test, t2_y_test)[1]))
            predictions.append(':'.join([str(i) for i in \
                predictions_matrix(model, t1_x_test, t1_y_test) + predictions_matrix(model, t2_x_test, t2_y_test)]))
        print('{},{},post,{},{},{},{},{}'.format(
            print_prefix,
            t1_epoch,
            '_'.join(t1_accuracies),
            '_'.join(t2_accuracies),
            t1_accuracies[-1],
            t2_accuracies[-1],
            '_'.join(predictions)),
            file=outfile)
        model.load_weights(weights_filename)
        t1_accuracies = t1_accuracies[:t1_epoch]
        t2_accuracies = t2_accuracies[:t1_epoch]
        predictions = predictions[:t1_epoch]
os.remove(weights_filename)

# close results file
outfile.close()
