#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import warnings

# parse args
parser = argparse.ArgumentParser(
    description='This is the main mnist and fashion mnist experiment.')
parser.add_argument(
    'outfile',
    type=str,
    help='json file to dump results to; will terminate if file already exists')
parser.add_argument(
    'dataset',
    type=str,
    choices=['mnist', 'fashion_mnist'],
    help='dataset to use in experiments')
parser.add_argument(
    'folds',
    type=int,
    help='number of folds contained in masks')
parser.add_argument(
    'validation_fold',
    type=int,
    help='fold index to use for determining when accuracy criteria is met')
parser.add_argument(
    'test_fold',
    type=int,
    help='fold index for testing')
parser.add_argument(
    'phases',
    type=str,
    help='colon separated sets of digits for different phases')
parser.add_argument(
    'criteria',
    type=float,
    help='accuracy criteria to move onto next stage')
parser.add_argument(
    'tolerance',
    type=int,
    help='maximum number of epochs to try and satisfy criteria in a single phase')
parser.add_argument(
    'architecture',
    type=str,
    help='colon separated architecture of the hidden layers of the network')
parser.add_argument(
    'seed',
    type=int,
    help='seed for numpy random number generator; used for network initialization')
subparsers = parser.add_subparsers(
    dest='optimizer',
    help='optimizer for training')
sgd_parser = subparsers.add_parser('sgd')
adam_parser = subparsers.add_parser('adam')
rms_parser = subparsers.add_parser('rms')
for subparser in [sgd_parser, adam_parser, rms_parser]:
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
experiment = vars(parser.parse_args())

# check args
if os.path.isfile(experiment['outfile']):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)
assert(0 < experiment['folds'])
assert(0 <= experiment['validation_fold'] < experiment['folds'])
assert(0 <= experiment['test_fold'] < experiment['folds'])
assert(experiment['validation_fold'] != experiment['test_fold'])
experiment['phases'] = [[int(i) for i in x] for x in experiment['phases'].split(':')]
experiment['digits'] = sorted(list(set(i for j in experiment['phases'] for i in j)))
assert(all([0 <= digit <= 9 for digit in experiment['digits']]))
experiment['has_all_digits_phase'] = tuple(experiment['digits']) in \
    set([tuple(phase) for phase in experiment['phases']])
assert(0 < experiment['criteria'] <= 1)
architecture = [int(i) for i in experiment['architecture'].split(':')]
assert(all([0 < i for i in architecture]))
if 'momentum' not in experiment:
    experiment['momentum'] = None
if 'beta_1' not in experiment:
    assert('beta_2' not in experiment)
    experiment['beta_1'] = None
    experiment['beta_2'] = None
if 'rho' not in experiment:
    experiment['rho'] = None

# args processed; import everything for experiment
import json
import numpy as np
import tensorflow as tf

# setup libraries
tf.logging.set_verbosity('FATAL')
warnings.filterwarnings('ignore')
np.random.seed(experiment['seed'])

# load dataset and masks
if experiment['dataset'] == 'mnist':
    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = \
        tf.keras.datasets.mnist.load_data()
    masks = np.load('mnist_masks.npy', allow_pickle=True)
else:
    assert experiment['dataset'] == 'fashion_mnist'
    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = \
        tf.keras.datasets.fashion_mnist.load_data()
    masks = np.load('fashion_mnist_masks.npy', allow_pickle=True)
raw_x_train, raw_x_test = raw_x_train / 255.0, raw_x_test / 255.0

# process masks
assert(masks.shape[0] > experiment['folds'])  # make sure we're not touching the holdout
def build_masks(digits, validation_fold, test_fold):
    train_mask = np.zeros(len(raw_y_train), dtype=bool)
    validation_mask = np.copy(train_mask)
    test_mask = np.copy(train_mask)
    for digit in digits:
        for fold in range(experiment['folds']):
            if fold == test_fold:
                test_mask += masks[fold][digit]
            elif fold == validation_fold:
                validation_mask += masks[fold][digit]
            else:
                train_mask += masks[fold][digit]
    return train_mask, validation_mask, test_mask
train_masks, validation_masks, test_masks = list(), list(), list()
if experiment['has_all_digits_phase']:
    phases = experiment['phases']
else:
    phases = experiment['phases'] + [experiment['digits']]
for phase in phases:
    train_mask, validation_mask, test_mask = \
        build_masks(phase, experiment['validation_fold'], experiment['test_fold'])
    train_masks.append(train_mask)
    validation_masks.append(validation_mask)
    test_masks.append(test_mask)

# build train and test datasets
x_train, y_train = list(), list()
x_validation, y_validation = list(), list()
x_test, y_test = list(), list()
for i in range(len(train_masks)):
    x_train.append(raw_x_train[train_masks[i], ...])
    y_train.append(raw_y_train[train_masks[i]] - min(experiment['digits']))
    x_validation.append(raw_x_train[validation_masks[i], ...])
    y_validation.append(raw_y_train[validation_masks[i]] - min(experiment['digits']))
    x_test.append(raw_x_train[test_masks[i], ...])
    y_test.append(raw_y_train[test_masks[i]] - min(experiment['digits']))

# build model
layers = list()
layers.append(tf.keras.layers.Flatten(input_shape=(28, 28)))
for hidden_units in architecture:
    layers.append(tf.keras.layers.Dense(
        hidden_units,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            seed=np.random.randint(2 ** 16 - 1))))
layers.append(tf.keras.layers.Dense(
    len(experiment['digits']),
    activation=tf.nn.softmax,
    kernel_initializer=tf.keras.initializers.RandomNormal(
        seed=np.random.randint(2 ** 16 - 1))))
model = tf.keras.models.Sequential(layers)

# prepare buffers to store results
experiment['phase_length'] = list()
experiment['accuracies'] = list()
experiment['predictions'] = list()
experiment['success'] = True

# prepare optimizer
if experiment['optimizer'] == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        lr=float(experiment['lr']),
        momentum=float(experiment['momentum']))
elif experiment['optimizer'] == 'rms':
    optimizer = tf.keras.optimizers.RMSprop(
        lr=float(experiment['lr']),
        rho=float(experiment['rho']))
else:
    assert(experiment['optimizer'] == 'adam')
    optimizer = tf.keras.optimizers.Adam(
        lr=float(experiment['lr']),
        beta_1=float(experiment['beta_1']),
        beta_2=float(experiment['beta_2']))

# create helper functions
def get_accuracies():
    rv = list()
    for phase in range(len(x_test)):
        rv.append(float(model.evaluate(x_test[phase], y_test[phase])[1]))
    return rv
def get_predictions():
    rv = np.zeros((len(experiment['digits']), len(experiment['digits'])), dtype=float)
    predictions = np.argmax(model.predict(x_test[-1]), axis=1)
    for i in range(len(experiment['digits'])):
        for j in range(len(experiment['digits'])):
            count = np.sum(np.logical_and(
                predictions + min(experiment['digits']) == experiment['digits'][i],
                y_test[-1] + min(experiment['digits']) == experiment['digits'][j]))
            rv[i, j] = \
                count / np.sum(y_test[-1] + min(experiment['digits']) == experiment['digits'][j])
    return rv.tolist()
def validate(phase):
    return model.evaluate(x_validation[phase], y_validation[phase])[1] < experiment['criteria']

# run experiment
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
experiment['accuracies'].append(get_accuracies())
experiment['predictions'].append(get_predictions())
for phase in range(len(experiment['phases'])):
    step = 0
    while validate(phase):
        model.fit(x_train[phase], y_train[phase])
        experiment['accuracies'].append(get_accuracies())
        experiment['predictions'].append(get_predictions())
        step += 1
        if step > experiment['tolerance']:
            experiment['success'] = False
            break
    if experiment['success']:
        experiment['phase_length'].append(step)
    else:
        break

# save results
assert(not os.path.isfile(experiment['outfile']))
with open(experiment['outfile'], 'w') as outfile:
    json.dump(experiment, outfile, sort_keys=True)
