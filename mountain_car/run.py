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
    help='number of episodes to run for',
    required=True)
parser.add_argument(
    '--env-seed',
    type=int,
    help='seed for episode initialization',
    required=False)
parser.add_argument(
    '--network-seed',
    type=int,
    help='seed for network initialization',
    required=False)
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
         'requried by all optimizers',
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
if experiment['optimizer'] == 'nmom':
    assert(experiment['momentum'] is not None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is None)

# args ok; start experiment
experiment['start_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

# setup libraries
warnings.filterwarnings('ignore')
if experiment['network_seed'] is not None:
    torch.manual_seed(experiment['network_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# setup neural network
dtype = torch.float
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
else:
    assert(experiment['optimizer'] == 'adam')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment['lr'],
        betas=(experiment['beta_1'], experiment['beta_2'])
    )

# load test states
test_data = np.load('test_states.npz')
x_test = torch.tensor(test_data['x'], dtype=dtype)
y_test = torch.tensor(test_data['y'], dtype=dtype)

# load interference test states
interference_test_data = np.load('interference_test_states.npz')
interference_x_test = torch.tensor(interference_test_data['x'], dtype=dtype)
interference_y_test = torch.tensor(interference_test_data['y'], dtype=dtype)

# prepare buffers to store results
experiment['steps'] = list()
experiment['accuracy'] = list()
experiment['pairwise_interference'] = list()
experiment['activation_overlap'] = list()

# prepare environment
if experiment['env_seed'] is None:
    env = MountainCarPrediction()
else:
    env = MountainCarPrediction(
        generator=np.random.RandomState(experiment['env_seed']))

# build helper function to compute test error and interference
def test(model):
    with torch.no_grad():
        return ((y_test - model(x_test)) ** 2).mean().sqrt().item()
def test_pairwise_interference(model, loss):
    grads = list()
    for i in range(len(interference_x_test)):
        y_pred = model(interference_x_test[i])
        loss = loss_fn(y_pred, interference_y_test[i])
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
def test_activation_overlap(model):
    activations = list()
    with torch.no_grad():
        for i in range(len(interference_x_test)):
            activations.append(
                (list(model.children())[0].forward(interference_x_test[i]) > 0).numpy())
    mean, count = 0, 0
    for i in range(len(activations)):
        for j in range(i, len(activations)):
            value = np.mean(np.logical_and(activations[i], activations[j]))
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
    experiment['steps'].append(step)

    # calculate returns
    returns = list()
    for i, observation in enumerate(observations):
        returns.append(- (len(observations) - i - 2))

    # train network online on episode
    for (position, velocity), return_ in zip(observations, returns):
        with torch.no_grad():
            x[0] = position
            x[1] = velocity
            y[0] = return_
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    experiment['accuracy'].append(test(model))
    experiment['pairwise_interference'].append(test_pairwise_interference(model, loss))
    experiment['activation_overlap'].append(test_activation_overlap(model))

# save results
assert(not os.path.isfile(experiment['outfile']))
experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
with open(experiment['outfile'], 'w') as outfile:
    json.dump(experiment, outfile, sort_keys=True)
