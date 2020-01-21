#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCarPrediction
from tools import *
import argparse
import datetime
import json
import numpy as np
import os
import sys
import torch
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
    '--num-episodes',
    type=int,
    help='number of episodes to run for; '
         'required',
    required=True)
parser.add_argument(
    '--env-seed',
    type=int,
    help='seed for episode initialization',
    default=None)
parser.add_argument(
    '--network-seed',
    type=int,
    help='seed for network initialization; '
         'not permitted if optimizer is set to constant',
    default=None)
parser.add_argument(
    '--optimizer',
    choices=['constant', 'sgd', 'adam', 'rms'],
    help='optimization algorithm to use to train the network; '
         'if set to constant than a constant predictor rather is used instead of a network; '
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
parser.add_argument(
    '--loss',
    choices=['squared_error', 'TD'],
    help='loss function to use when training the network; '
         'requried by sgd, adam, and rms optimizer only',
    default=None)
experiment = vars(parser.parse_args())

# check that the outfile doesn't already exist
if os.path.isfile(experiment['outfile']):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)

# check that optimizer arguments are specified correctly
if experiment['optimizer'] == 'constant':
    assert(experiment['network_seed'] is None)
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is None)
    assert(experiment['loss'] is None)
if experiment['optimizer'] == 'sgd':
    assert(experiment['momentum'] is not None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is None)
    assert(experiment['loss'] is not None)
if experiment['optimizer'] == 'adam':
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is not None)
    assert(experiment['beta_2'] is not None)
    assert(experiment['rho'] is None)
    assert(experiment['loss'] is not None)
if experiment['optimizer'] == 'rms':
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is not None)
    assert(experiment['loss'] is not None)

# args ok; start experiment
experiment['start_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

# setup libraries
warnings.filterwarnings('ignore')
if experiment['network_seed'] is not None:
    torch.manual_seed(experiment['network_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# setup predictor
dtype = torch.float
if experiment['optimizer'] == 'constant':
    class ConstantPredictor:
        def __init__(self):
            self.mean = 0
            self.count = 0

        def __call__(self, x):
            try:
                size = len(x)
            except TypeError:
                size = 1
            return torch.ones(size, requires_grad=False) * self.mean

        def update(self, return_):
            self.count += 1
            self.mean += (return_ - self.mean) / self.count
    model = ConstantPredictor()
else:
    linear1 = torch.nn.Linear(2, 50)
    torch.nn.init.normal_(linear1.weight, std=0.05)
    relu1 = torch.nn.ReLU()
    linear2 = torch.nn.Linear(50, 1)
    torch.nn.init.normal_(linear2.weight, std=0.05)
    model = torch.nn.Sequential(
        linear1,
        relu1,
        linear2
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    x = torch.empty(2, dtype=dtype)
    y = torch.empty(1, dtype=dtype)

# setup optimizer
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
elif experiment['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment['lr'],
        betas=(experiment['beta_1'], experiment['beta_2'])
    )
else:
    assert(experiment['optimizer'] == 'constant')

# load test states
test_data = np.load('test_states.npz')
test_data_x = torch.tensor(test_data['x'], dtype=dtype)
test_data_y = torch.tensor(test_data['y'], dtype=dtype)
x_test = torch.empty(2, dtype=dtype)
y_test = torch.empty(1, dtype=dtype)

# load interference test states
interference_test_data = np.load('interference_test_states.npz')
interference_x_test = torch.tensor(interference_test_data['x'], dtype=dtype)
interference_y_test = torch.tensor(interference_test_data['y'], dtype=dtype)

# prepare buffers to store results
experiment['steps'] = list()
experiment['accuracy'] = list()
if experiment['optimizer'] != 'constant':
    experiment['pairwise_interference'] = list()
    experiment['activation_overlap'] = list()

# prepare environment
if experiment['env_seed'] is None:
    env = MountainCarPrediction()
else:
    env = MountainCarPrediction(
        generator=np.random.RandomState(experiment['env_seed']))

# build helper function to compute test error and interference
def test():
    with torch.no_grad():
        return ((test_data_y - model(test_data_x)) ** 2).mean().sqrt().item()
def test_pairwise_interference():
    grads = list()
    for i in range(len(interference_x_test)):
        with torch.no_grad():
            position = interference_x_test[i][0]
            velocity = interference_x_test[i][1]
            return_ = interference_y_test[i]
            next_position, next_velocity = MountainCarPrediction.get_next_observation((position,
                                                                                       velocity))
            next_return = MountainCarPrediction.get_return((next_position,
                                                            next_velocity))
            if experiment['loss'] == 'squared_error':
                x_test[0] = position
                x_test[1] = velocity
                y_test[0] = return_
            else:
                assert(experiment['loss'] == 'TD')
                x_test[0] = next_position
                x_test[1] = next_velocity
                y_test[0] = next_return - return_ + model(x_test)
                x_test[0] = position
                x_test[1] = velocity
        y_pred = model(x_test)
        loss = loss_fn(y_pred, y_test)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grads.append(torch.cat([i.grad.flatten() for i in model.parameters()]).numpy())
    mean, count = 0, 0
    for i in range(len(grads)):
        for j in range(i, len(grads)):
            value = grads[i].dot(grads[j])
            value /= np.sqrt(grads[i].dot(grads[i])) * np.sqrt(grads[j].dot(grads[j]))
            count += 1
            mean += (value - mean) / count
    return mean
def test_activation_overlap():
    activations = list()
    with torch.no_grad():
        for i in range(len(interference_x_test)):
            activations.append((linear1.forward(interference_x_test[i]) > 0).numpy())
    mean, count = 0, 0
    for i in range(len(activations)):
        for j in range(i, len(activations)):
            value = np.mean(np.logical_and(activations[i], activations[j]))
            # print('EPISODE: {0:3}, i: {1:2}, j: {2:2}, ACTIVATIONS[i]: {3}, ACTIVATIONS[j]: {4}, ACTIVATIONS[i & j]: {5}, MEAN: {6}'.format(
            #     episode,
            #     i,
            #     j,
            #     ''.join(['1' if i else '0' for i in activations[i]]),
            #     ''.join(['1' if i else '0' for i in activations[j]]),
            #     ''.join(['1' if i else '0' for i in np.logical_and(activations[i], activations[j])]),
            #     value))
            count += 1
            mean += (value - mean) / count
    return mean

# run experiment
all_observations = list()
all_returns = list()
for episode in range(experiment['num_episodes']):
    observation = env.reset()
    observations = list()
    done = False
    step = 0
    while not done:
        observations.append(observation)
        observation, reward, done = env.step()
        step += 1
    terminal_observation = observation
    experiment['steps'].append(step)

    # calculate returns
    returns = list()
    for i, observation in enumerate(observations):
        returns.append(- (len(observations) - i - 2))

    # train network online on episode
    for i in range(len(returns)):
        position = observations[i][0]
        velocity = observations[i][1]
        return_ = returns[i]
        try:
            next_position = observations[i + 1][0]
            next_velocity = observations[i + 1][1]
        except IndexError:
            next_position = terminal_observation[0]
            next_velocity = terminal_observation[1]
        try:
            next_return = returns[i + 1]
        except IndexError:
            next_return = 0
        if experiment['optimizer'] == 'constant':
            model.update(return_)
        else:
            with torch.no_grad():
                if experiment['loss'] == 'squared_error':
                    x[0] = position
                    x[1] = velocity
                    y[0] = return_
                else:
                    assert(experiment['loss'] == 'TD')
                    x[0] = next_position
                    x[1] = next_velocity
                    y[0] = next_return - return_ + model(x)
                    x[0] = position
                    x[1] = velocity
            y_pred = model(x)
            # with torch.no_grad():
            #     print('EPISODE: {0:3}, position: {1:7.4f}, velocity: {2:7.4f}, y: {3:9.4f}, y_pred: {4:9.4f}, loss: {5:10.4f}'.format(
            #         episode,
            #         x[0].item(),
            #         x[1].item(),
            #         y.item(),
            #         y_pred.item(),
            #         loss_fn(y_pred, y).item()))
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    experiment['accuracy'].append(test())
    if experiment['optimizer'] != 'constant':
        experiment['pairwise_interference'].append(test_pairwise_interference())
        experiment['activation_overlap'].append(test_activation_overlap())

# save results
assert(not os.path.isfile(experiment['outfile']))
experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
with open(experiment['outfile'], 'w') as outfile:
    json.dump(experiment, outfile, sort_keys=True)
