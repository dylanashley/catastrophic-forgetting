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
    help='json file to dump results to; will terminate if file already exists; '
         'required',
    required=True)
parser.add_argument(
    '--dataset',
    choices=['mnist', 'fashion_mnist'],
    help='dataset to use in experiments; '
         'required',
    required=True)
parser.add_argument(
    '--fold-count',
    type=int,
    help='number of folds contained in masks; '
         'required',
    required=True)
parser.add_argument(
    '--train-folds',
    type=str,
    help='colon separated set of folds to use for training; '
         'can also specify the training folds to use in each phase using double colons; '
         'required',
    required=True)
parser.add_argument(
    '--prevent-repeats',
    action='store_true',
    help='prevent the same sample appearing multiple times in the same phase')
parser.add_argument(
    '--test-folds',
    type=str,
    help='colon separated set of folds to use for testing')
parser.add_argument(
    '--phases',
    type=str,
    help='colon separated sets of digits for different phases; '
         'required',
    required=True)
parser.add_argument(
    '--test-on-all-digits',
    action='store_true',
    help='also test on all the digits appearing in any phase; '
         'requires test folds to be specified')
parser.add_argument(
    '--log-frequency',
    type=int,
    help='number of examples to learn on between recording accuracies and predictions; '
         'required',
    required=True)
parser.add_argument(
    '--init-seed',
    type=int,
    help='seed for network initialization; '
         'required',
    required=True)
parser.add_argument(
    '--shuffle-seed',
    type=int,
    help='seed for the random number generator used to shuffle datasets')
parser.add_argument(
    '--criteria',
    choices=['steps', 'offline', 'online'],
    help='type of criteria to use when deciding whether or not to move to the next phase')
parser.add_argument(
    '--steps',
    type=int,
    help='number of steps in each phase; '
         'required for steps criteria only',
    default=None)
parser.add_argument(
    '--required-accuracy',
    type=float,
    help='required classifier accuracy to move onto next phase; '
         'required for offline and online criteria only',
    default=None)
parser.add_argument(
    '--tolerance',
    type=int,
    help='maximum number of steps to try and satisfy required accuracy in a single phase; '
         'required for offline and online criteria only',
    default=None)
parser.add_argument(
    '--validation-folds',
    type=str,
    help='colon separated set of folds to use for determining when accuracy criteria is met; '
         'required for offline criteria only',
    default=None)
parser.add_argument(
    '--minimum-steps',
    type=int,
    help='minimum number of steps before moving onto next phase; '
         'required for online criteria only',
    default=None)
parser.add_argument(
    '--hold-steps',
    type=int,
    help='minimum number of steps to hold accuracy for before moving onto next stage; '
         'required for online criteria only',
    default=None)
parser.add_argument(
    '--optimizer',
    choices=['sgd', 'adam', 'rms'],
    help='optimization algorithm to use to train the network; '
         'required',
    required=True)
parser.add_argument(
    '--lr',
    type=float,
    help='learning rate for training; '
         'requried by sgd, adam, and rms optimizer only',
    default=None)
parser.add_argument(
    '--momentum',
    type=float,
    help='momentum hyperparameter; '
         'requried by sgd optimizer only',
    default=None)
parser.add_argument(
    '--beta-1',
    type=float,
    help='beta 1 hyperparameter; '
         'requried by adam optimizer only',
    default=None)
parser.add_argument(
    '--beta-2',
    type=float,
    help='beta 2 hyperparameter; '
         'requried by adam optimizer only',
    default=None)
parser.add_argument(
    '--rho',
    type=float,
    help='rho hyperparameter; '
         'required by rms optimizer only',
    default=None)
experiment = vars(parser.parse_args())

# check that the outfile doesn't already exist
if os.path.isfile(experiment['outfile']):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)

# check and process fold and phase structure arguments are specified correctly
assert(0 < experiment['fold_count'])
experiment['phases'] = [[int(i) for i in x] for x in experiment['phases'].split(':')]
experiment['digits'] = sorted(list(set(i for j in experiment['phases'] for i in j)))
assert(all([0 <= digit <= 9 for digit in experiment['digits']]))
experiment['train_folds'] = experiment['train_folds'].split('::')
if len(experiment['train_folds']) != len(experiment['phases']):
    assert(len(experiment['train_folds']) == 1)
    experiment['train_folds'] = [experiment['train_folds'][0] for _ in experiment['phases']]
experiment['train_folds'] = [sorted([int(j)for j in folds.split(':')])
                             for folds in experiment['train_folds']]
assert(len(experiment['train_folds']) == len(experiment['phases']))
assert(all([0 <= i < experiment['fold_count'] for i in sum(experiment['train_folds'], [])]))
if experiment['test_folds'] is not None:
    experiment['test_folds'] = sorted([int(i)for i in experiment['test_folds'].split(':')])
    assert(all([0 <= i < experiment['fold_count'] for i in experiment['test_folds']]))
    assert(set(experiment['test_folds']).isdisjoint(sum(experiment['train_folds'], [])))
if experiment['test_on_all_digits']:
    assert(experiment['test_folds'] is not None)

# check and process criteria arguments
if experiment['criteria'] == 'steps':
    assert(experiment['steps'] is not None)
    assert(experiment['required_accuracy'] is None)
    assert(experiment['tolerance'] is None)
    assert(experiment['validation_folds'] is None)
    assert(experiment['minimum_steps'] is None)
    assert(experiment['hold_steps'] is None)
    assert(0 < experiment['steps'])
if experiment['criteria'] == 'offline':
    assert(experiment['steps'] is None)
    assert(experiment['required_accuracy'] is not None)
    assert(experiment['tolerance'] is not None)
    assert(experiment['validation_folds'] is not None)
    assert(experiment['minimum_steps'] is None)
    assert(experiment['hold_steps'] is None)
    assert(0 < experiment['required_accuracy'] <= 1)
    assert(0 < experiment['tolerance'])
    experiment['validation_folds'] = \
        sorted([int(i)for i in experiment['validation_folds'].split(':')])
    assert(all([0 <= i < experiment['fold_count'] for i in experiment['validation_folds']]))
    assert(set(experiment['validation_folds']).isdisjoint(sum(experiment['train_folds'], [])))
    assert(set(experiment['validation_folds']).isdisjoint(experiment['test_folds']))
if experiment['criteria'] == 'online':
    assert(experiment['steps'] is None)
    assert(experiment['required_accuracy'] is not None)
    if not experiment['prevent_repeats']:
        assert(experiment['tolerance'] is not None)
        assert(0 < experiment['tolerance'])
    assert(experiment['validation_folds'] is None)
    assert(experiment['minimum_steps'] is not None)
    assert(experiment['hold_steps'] is not None)
    assert(0 < experiment['required_accuracy'] <= 1)
    assert(0 <= experiment['minimum_steps'])
    assert(0 <= experiment['hold_steps'])

# check that optimizer arguments are specified correctly
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

# check and process all other arguments
assert(0 < experiment['log_frequency'])

# args ok; start experiment
experiment['start_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

# stop the annoying deprecation warnings when loading tensorflow
warnings.filterwarnings('ignore')

import json
import numpy as np
import tensorflow as tf
import torch

# setup libraries
torch.set_num_threads(1)
try:
    tf.logging.set_verbosity('FATAL')
except AttributeError:
    pass
if experiment['init_seed'] is not None:
    torch.manual_seed(experiment['init_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    return np.array(sum([masks[fold][digit] for fold in folds for digit in digits]), dtype=bool)
train_masks = [build_mask(phase, folds)
               for phase, folds in zip(experiment['phases'], experiment['train_folds'])]
if experiment['test_folds'] is not None:
    test_masks = [build_mask(phase, experiment['test_folds'])
                  for phase in experiment['phases']]
    if experiment['test_on_all_digits']:
        test_masks.append(build_mask(experiment['digits'], experiment['test_folds']))
if experiment['validation_folds'] is not None:
    validation_masks = [build_mask(phase, experiment['validation_folds'])
                        for phase in experiment['phases']]

# build datasets
def shuffle_jointly(x, y):
    z = list(zip(x, y))
    np.random.RandomState(seed=experiment['shuffle_seed']).shuffle(z)
    x, y = zip(* z)
    return x, y
x_train = [raw_x_train[mask, ...]
           for mask in train_masks]
y_train = [raw_y_train[mask, ...] - min(experiment['digits'])
           for mask in train_masks]
for i in range(len(x_train)):
    x_train[i], y_train[i] = shuffle_jointly(x_train[i], y_train[i])
    x_train[i] = torch.tensor(x_train[i], dtype=torch.float).flatten(start_dim=1)
    y_train[i] = torch.tensor(y_train[i], dtype=torch.int)
if experiment['test_folds'] is not None:
    x_test = [raw_x_train[mask, ...]
              for mask in test_masks]
    y_test = [raw_y_train[mask, ...] - min(experiment['digits'])
              for mask in test_masks]
    mask = np.zeros(len(y_test[-1]), dtype=bool)
    digit_counts = np.zeros(max(raw_y_train))
    for i, digit in enumerate(y_test[-1]):
        if (digit in experiment['digits']) and (digit_counts[digit] < 10):
            mask[i] = True
            digit_counts[digit] += 1
    x_ten_test = x_test[-1][mask, ...]  # smaller dataset for second order tests
    y_ten_test = y_test[-1][mask, ...]
    for i in range(len(x_test)):
        x_test[i] = torch.tensor(x_test[i], dtype=torch.float).flatten(start_dim=1)
        y_test[i] = torch.tensor(y_test[i], dtype=torch.int)
    x_ten_test = torch.tensor(x_ten_test, dtype=torch.float).flatten(start_dim=1)
    y_ten_test = torch.tensor(y_ten_test, dtype=torch.int)
if experiment['validation_folds'] is not None:
    x_validation = [raw_x_train[mask, ...]
                    for mask in validation_masks]
    y_validation = [raw_y_train[mask, ...] - min(experiment['digits'])
                    for mask in validation_masks]
    for i in range(len(x_validation)):
        x_validation[i] = torch.tensor(x_validation[i], dtype=torch.float).flatten(start_dim=1)
        y_validation[i] = torch.tensor(y_validation[i], dtype=torch.int)

# build model
dtype = torch.float
linear1 = torch.nn.Linear(28 * 28, 100)
torch.nn.init.normal_(linear1.weight, std=0.05)
relu1 = torch.nn.ReLU()
linear2 = torch.nn.Linear(100, len(experiment['digits']))
torch.nn.init.normal_(linear2.weight, std=0.05)
model = torch.nn.Sequential(
    linear1,
    relu1,
    linear2
)
loss_fn = torch.nn.CrossEntropyLoss()

# prepare optimizer
if experiment['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=experiment['lr'],
        momentum=experiment['momentum']
    )
elif experiment['optimizer'] == 'rms':
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=experiment['lr'],
        alpha=experiment['rho']
    )
else:
    assert(experiment['optimizer'] == 'adam')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment['lr'],
        betas=(experiment['beta_1'], experiment['beta_2'])
    )

# prepare buffers to store results
experiment['success'] = True
experiment['correct'] = list()
experiment['phase_length'] = list()
experiment['accuracies'] = None if experiment['test_folds'] is None else list()
experiment['predictions'] = None if experiment['test_folds'] is None else list()
experiment['pairwise_interference'] = None if experiment['test_folds'] is None else list()
experiment['activation_overlap'] = None if experiment['test_folds'] is None else list()

# create helper functions for metrics
@torch.no_grad()
def get_accuracies(model):
    rv = list()
    for phase in range(len(x_test)):
        x = x_test[phase]
        y_pred = model(x).argmax(axis=1)
        y = y_test[phase]
        accuracy = (y_pred == y).int().float().mean().item()
        rv.append(accuracy)
    return rv

@torch.no_grad()
def get_predictions(model):
    rv = torch.zeros((len(experiment['digits']), len(experiment['digits'])), dtype=torch.float)
    predictions = model(x_test[-1]).argmax(axis=1)
    for i in range(len(experiment['digits'])):
        for j in range(len(experiment['digits'])):
            y_pred = predictions + min(experiment['digits']) == experiment['digits'][i]
            y = y_test[-1] + min(experiment['digits']) == experiment['digits'][j]
            rv[i, j] = ((y_pred & y).int().sum().float() / y.sum().float()).item()
    return rv.tolist()

def get_pairwise_interference(model, loss_fn):
    grads = list()
    for i in range(len(x_ten_test)):
        x = x_ten_test[i]
        y_pred = model(x).double()
        y = y_ten_test[i].long()
        loss = loss_fn(y_pred.unsqueeze(0), y.unsqueeze(0))
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grads.append(torch.cat([i.grad.flatten() for i in model.parameters()]).numpy())

    # calculate pairwise interference
    same_mean = np.array([0 for _ in range(len(x_ten_test))], dtype=float)
    same_count = np.copy(same_mean)
    different_mean = np.copy(same_mean)
    different_count = np.copy(same_mean)
    for i in range(len(x_ten_test)):
        for j in range(i, len(x_ten_test)):
            value = grads[i].dot(grads[j])
            value /= np.sqrt(grads[i].dot(grads[i])) * np.sqrt(grads[j].dot(grads[j]))
            if y_ten_test[i] == y_ten_test[j]:
                same_count[i] += 1
                same_mean[i] += (value - same_mean[i]) / same_count[i]
            else:
                different_count[i] += 1
                different_mean[i] += (value - different_mean[i]) / different_count[i]
    return np.mean(different_mean / same_mean)

@torch.no_grad()
def get_activation_overlap(model):
    activations = list()
    for i in range(len(x_ten_test)):
        activations.append((linear1.forward(x_ten_test[i]) > 0))

    # calculate pairwise interference
    same_mean = np.array([0 for _ in range(len(x_ten_test))], dtype=float)
    same_count = np.copy(same_mean)
    different_mean = np.copy(same_mean)
    different_count = np.copy(same_mean)
    for i in range(len(x_ten_test)):
        for j in range(i, len(x_ten_test)):
            value = (activations[i] & activations[j]).int().float().mean().item()
            if y_ten_test[i] == y_ten_test[j]:
                same_count[i] += 1
                same_mean[i] += (value - same_mean[i]) / same_count[i]
            else:
                different_count[i] += 1
                different_mean[i] += (value - different_mean[i]) / different_count[i]
    return np.mean(different_mean / same_mean)

# create helper function for phase transitions
if experiment['criteria'] == 'steps':
    def phase_over(phase, step, number_correct, steps_since_last_error):
        return step == experiment['steps']
if experiment['criteria'] == 'offline':
    def phase_over(phase, step, number_correct, steps_since_last_error):
        return model.evaluate(x_validation[phase], y_validation[phase])[1] >= \
            experiment['required_accuracy']
if experiment['criteria'] == 'online':
    def phase_over(phase, step, number_correct, steps_since_last_error):
        return (step >= experiment['minimum_steps']) and \
            (steps_since_last_error >= experiment['hold_steps']) and \
            (number_correct / step >= experiment['required_accuracy'])

# run experiment
warned_about_repeats = False
for phase in range(len(experiment['phases'])):
    examples_in_phase = y_train[phase].shape[0]
    i = j = k = 0
    assert(not phase_over(phase, i, j, k))
    while True:
        if i > examples_in_phase:
            if experiment['prevent_repeats']:
                experiment['success'] = False
                break
            if not warned_about_repeats:
                warnings.warn('sampling examples with replacement')
                warned_about_repeats = True
        x = x_train[phase][i % examples_in_phase]
        y_pred = model(x).double()
        y = y_train[phase][i % examples_in_phase].long()
        with torch.no_grad():
            correct = bool(y_pred.argmax() == y)
            experiment['correct'].append(correct)
        loss = loss_fn(y_pred.unsqueeze(0), y.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (experiment['test_folds'] is not None) and (not (i % experiment['log_frequency'])):
            experiment['accuracies'].append(get_accuracies(model))
            experiment['predictions'].append(get_predictions(model))
            experiment['pairwise_interference'].append(get_pairwise_interference(model, loss_fn))
            experiment['activation_overlap'].append(get_activation_overlap(model))
        i += 1
        if correct:
            j += 1
            k += 1
        else:
            k = 0
        if phase_over(phase, i, j, k):
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
experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
with open(experiment['outfile'], 'w') as outfile:
    json.dump(experiment, outfile, sort_keys=True)
