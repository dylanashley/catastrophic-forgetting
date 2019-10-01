#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import sys
import warnings

# parse args
parser = argparse.ArgumentParser(
    description='This is the main mnist and fashion mnist experiment.')
parser.add_argument(
    '--outfile',
    type=str,
    help='json file to dump results to; will terminate if file already exists',
    required=True)
parser.add_argument(
    '--dataset',
    choices=['mnist', 'fashion_mnist'],
    help='dataset to use in experiments',
    required=True)
parser.add_argument(
    '--fold-count',
    type=int,
    help='number of folds contained in masks',
    required=True)
parser.add_argument(
    '--train-folds',
    type=str,
    help='colon separated set of folds to use for training',
    required=True)
parser.add_argument(
    '--test-folds',
    type=str,
    help='colon separated set of folds to use for testing',
    required=True)
parser.add_argument(
    '--phases',
    type=str,
    help='colon separated sets of digits for different phases',
    required=True)
parser.add_argument(
    '--log-frequency',
    type=int,
    help='number of examples to learn on before recording accuracies and predictions',
    required=True)
parser.add_argument(
    '--architecture',
    type=str,
    help='colon separated architecture of the hidden layers of the network',
    required=True)
parser.add_argument(
    '--seed',
    type=int,
    help='seed for numpy random number generator; used for network initialization',
    required=True)
parser.add_argument(
    '--criteria',
    choices=['steps', 'offline', 'online'],
    help='type of criteria to use for deciding when to move to the next phase',
    required=True)
parser.add_argument(
    '--steps',
    type=int,
    help='number of steps in each phase; '
         'needed for steps criteria',
    default=None)
parser.add_argument(
    '--required-accuracy',
    type=float,
    help='accuracy criteria to move onto next stage; '
         'needed for offline and online criteria',
    default=None)
parser.add_argument(
    '--tolerance',
    type=int,
    help='maximum number of steps to try and satisfy required accuracy in a single phase; '
         'needed for offline and online criteria',
    default=None)
parser.add_argument(
    '--validation-folds',
    type=str,
    help='colon separated set of folds to use for determining when accuracy criteria is met; '
         'needed for offline criteria',
    default=None)
parser.add_argument(
    '--minimum-steps',
    type=int,
    help='minimum number of steps before moving onto next stage; '
         'needed for online criteria',
    default=None)
parser.add_argument(
    '--optimizer',
    choices=['sgd', 'adam', 'rms'],
    help='optimizer for training',
    required=True)
parser.add_argument(
    '--lr',
    type=float,
    help='learning rate for training; '
         'needed for sgd, adam, and rms optimizer',
    default=None)
parser.add_argument(
    '--momentum',
    type=float,
    help='momentum hyperparameter; '
         'needed for sgd optimizer',
    default=None)
parser.add_argument(
    '--beta-1',
    type=float,
    help='beta 1 hyperparameter; '
         'needed for adam optimizer',
    default=None)
parser.add_argument(
    '--beta-2',
    type=float,
    help='beta 2 hyperparameter; '
         'needed for adam optimizer',
    default=None)
parser.add_argument(
    '--rho',
    type=float,
    help='rho hyperparameter; '
         'needed for rms optimizer',
    default=None)
experiment = vars(parser.parse_args())

# check args
if os.path.isfile(experiment['outfile']):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)
assert(0 < experiment['fold_count'])
experiment['train_folds'] = sorted([int(i)for i in experiment['train_folds'].split(':')])
experiment['test_folds'] = sorted([int(i)for i in experiment['test_folds'].split(':')])
train_test_folds = experiment['train_folds'] + experiment['test_folds']
assert(all([0 <= i < experiment['fold_count'] for i in train_test_folds]))
assert(len(train_test_folds) == len(set(train_test_folds)))
experiment['phases'] = [[int(i) for i in x] for x in experiment['phases'].split(':')]
experiment['digits'] = sorted(list(set(i for j in experiment['phases'] for i in j)))
assert(all([0 <= digit <= 9 for digit in experiment['digits']]))
experiment['has_all_digits_phase'] = tuple(experiment['digits']) in \
    set([tuple(phase) for phase in experiment['phases']])
assert(0 < experiment['log_frequency'])
architecture = [int(i) for i in experiment['architecture'].split(':')]
assert(all([0 < i for i in architecture]))
if experiment['criteria'] == 'steps':
    assert(experiment['steps'] is not None)
    assert(experiment['required_accuracy'] is None)
    assert(experiment['tolerance'] is None)
    assert(experiment['validation_folds'] is None)
    assert(experiment['minimum_steps'] is None)
    assert(0 < experiment['steps'])
if experiment['criteria'] == 'offline':
    assert(experiment['steps'] is None)
    assert(experiment['required_accuracy'] is not None)
    assert(experiment['tolerance'] is not None)
    assert(experiment['validation_folds'] is not None)
    assert(experiment['minimum_steps'] is None)
    assert(0 < experiment['required_accuracy'] <= 1)
    assert(0 < experiment['tolerance'])
    experiment['validation_folds'] = \
        sorted([int(i)for i in experiment['validation_folds'].split(':')])
    all_folds = train_test_folds + experiment['validation_folds']
    assert(all([0 <= i < experiment['fold_count'] for i in all_folds]))
    assert(len(all_folds) == len(set(all_folds)))
if experiment['criteria'] == 'online':
    assert(experiment['steps'] is None)
    assert(experiment['required_accuracy'] is not None)
    assert(experiment['tolerance'] is not None)
    assert(experiment['validation_folds'] is None)
    assert(experiment['minimum_steps'] is not None)
    assert(0 < experiment['required_accuracy'] <= 1)
    assert(0 < experiment['tolerance'])
    assert(0 <= experiment['minimum_steps'])
if experiment['optimizer'] == 'sgd':
    assert(experiment['momentum'] is not None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is None)
if experiment['optimizer'] == 'adam':
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is not None)
    assert(experiment['beta_2'] is not None)
    assert(experiment['rho'] is None)
if experiment['optimizer'] == 'rms':
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is not None)

# args processed; start experiment
experiment['start_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

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
raw_x_test, raw_y_test = None, None  # disable use of the holdout dataset

# build masks
assert(masks.shape[0] > experiment['fold_count'])  # make sure we're not touching the holdout
def build_mask(digits, folds):
    mask = np.zeros(len(raw_y_train), dtype=bool)
    for digit in digits:
        for fold in folds:
            mask += masks[fold][digit]
    return mask
if experiment['has_all_digits_phase']:
    phases = experiment['phases']
else:
    phases = experiment['phases'] + [experiment['digits']]
train_masks = [build_mask(phase, experiment['train_folds']) for phase in phases]
test_masks = [build_mask(phase, experiment['test_folds']) for phase in phases]
if experiment['validation_folds'] is not None:
    validation_masks = [build_mask(phase, experiment['validation_folds']) for phase in phases]

# build datasets
x_train = [raw_x_train[mask, ...] for mask in train_masks]
y_train = [raw_y_train[mask, ...] - min(experiment['digits']) for mask in train_masks]
x_test = [raw_x_train[mask, ...] for mask in test_masks]
y_test = [raw_y_train[mask, ...] - min(experiment['digits']) for mask in test_masks]
if experiment['validation_folds'] is not None:
    x_validation = [raw_x_train[mask, ...]
                    for mask in validation_masks]
    y_validation = [raw_y_train[mask, ...] - min(experiment['digits'])
                    for mask in validation_masks]

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
experiment['correct'] = list()
experiment['success'] = True

# prepare optimizer
if experiment['optimizer'] == 'sgd':
    optimizer = tf.keras.optimizers.SGD(
        lr=experiment['lr'],
        momentum=experiment['momentum'])
elif experiment['optimizer'] == 'rms':
    optimizer = tf.keras.optimizers.RMSprop(
        lr=experiment['lr'],
        rho=experiment['rho'])
else:
    assert(experiment['optimizer'] == 'adam')
    optimizer = tf.keras.optimizers.Adam(
        lr=experiment['lr'],
        beta_1=experiment['beta_1'],
        beta_2=experiment['beta_2'])

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
if experiment['criteria'] == 'steps':
    def phase_over(phase, step, correct):
        return step == experiment['steps']
if experiment['criteria'] == 'offline':
    def phase_over(phase, step, correct):
        return model.evaluate(x_validation[phase], y_validation[phase])[1] >= \
            experiment['required_accuracy']
if experiment['criteria'] == 'online':
    def phase_over(phase, step, correct):
        return (step >= experiment['minimum_steps']) and \
            (correct / step >= experiment['required_accuracy'])

# run experiment
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
for phase in range(len(experiment['phases'])):
    examples_in_phase = y_train[phase].shape[0]
    i = 0
    j = 0
    assert(not phase_over(phase, i, j))
    while True:
        guess = int(np.argmax(
            model.predict(x_train[phase][i % examples_in_phase:i % examples_in_phase + 1])))
        correct = bool(guess == y_train[phase][i % examples_in_phase])
        j += correct
        experiment['correct'].append(correct)
        model.fit(x_train[phase][i % examples_in_phase:i % examples_in_phase + 1],
                  y_train[phase][i % examples_in_phase:i % examples_in_phase + 1])
        if not (i % experiment['log_frequency']):
            experiment['accuracies'].append(get_accuracies())
            experiment['predictions'].append(get_predictions())
        i += 1
        if phase_over(phase, i, j):
            experiment['phase_length'].append(i)
            experiment['success'] = True
            break
        if (experiment['tolerance'] is not None) and (i > experiment['tolerance']):
            experiment['success'] = False
            break
    if not experiment['success']:
        break

# save results
assert(not os.path.isfile(experiment['outfile']))
with open(experiment['outfile'], 'w') as outfile:
    experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    json.dump(experiment, outfile, sort_keys=True)
