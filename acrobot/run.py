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
    '--num-episodes',
    type=int,
    help='number of episodes to run for; '
         'required',
    required=True)
parser.add_argument(
    '--env-seed',
    type=int,
    help='seed for episode initialization')
parser.add_argument(
    '--approximator',
    choices=['constant', 'neural_network'],
    help='type of function approximator to use; '
         'if set to constant than a constant predictor is used instead of an approximator; '
         'required',
    required=True)
parser.add_argument(
    '--network-seed',
    type=int,
    help='seed for network initialization; '
         'useable with neural network approximator only')
parser.add_argument(
    '--loss',
    choices=['squared_error', 'TD'],
    help='loss function to use when training the network; '
         'required by neural network approximator only')
parser.add_argument(
    '--target-update',
    type=int,
    help='how often to update the target network; '
         'if set to 1 then no target network is used; '
         'required by TD loss function only')
parser.add_argument(
    '--optimizer',
    choices=['sgd', 'adam', 'rms'],
    help='optimization algorithm to use to train the network; '
         'required by neural network approximator only')
parser.add_argument(
    '--lr',
    type=float,
    help='learning rate for training; '
         'requried by tile coder approximator as well as sgd, adam, and rms optimizer only')
parser.add_argument(
    '--lambda',
    type=float,
    help='eligibility trace decay parameter; '
         'requried by tile coder approximator only')
parser.add_argument(
    '--momentum',
    type=float,
    help='momentum hyperparameter; '
         'requried by sgd optimizer only')
parser.add_argument(
    '--beta-1',
    type=float,
    help='beta 1 hyperparameter; '
         'requried by adam optimizer only')
parser.add_argument(
    '--beta-2',
    type=float,
    help='beta 2 hyperparameter; '
         'requried by adam optimizer only')
parser.add_argument(
    '--rho',
    type=float,
    help='rho hyperparameter; '
         'required by rms optimizer only')
experiment = vars(parser.parse_args())

# check that the outfile doesn't already exist
if (experiment['outfile'] != '-') and (os.path.isfile(experiment['outfile'])):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)

# check that the other arguments are specified correctly
if experiment['approximator'] == 'constant':
    assert(experiment['network_seed'] is None)
    assert(experiment['optimizer'] is None)
    assert(experiment['lr'] is None)
    assert(experiment['lambda'] is None)
    assert(experiment['momentum'] is None)
    assert(experiment['beta_1'] is None)
    assert(experiment['beta_2'] is None)
    assert(experiment['rho'] is None)
    assert(experiment['loss'] is None)
else:
    assert(experiment['approximator'] == 'neural_network')
    assert(experiment['loss'] is not None)
    if experiment['loss'] == 'TD':
        assert(experiment['target_update'] is not None)
        assert(experiment['target_update'] >= 1)
    else:
        assert(experiment['loss'] == 'squared_error')
        assert(experiment['target_update'] is None)
    assert(experiment['optimizer'] is not None)
    if experiment['optimizer'] == 'sgd':
        assert(experiment['lr'] is not None)
        assert(experiment['lambda'] is None)
        assert(experiment['momentum'] is not None)
        assert(experiment['beta_1'] is None)
        assert(experiment['beta_2'] is None)
        assert(experiment['rho'] is None)
    if experiment['optimizer'] == 'adam':
        assert(experiment['lr'] is not None)
        assert(experiment['lambda'] is None)
        assert(experiment['momentum'] is None)
        assert(experiment['beta_1'] is not None)
        assert(experiment['beta_2'] is not None)
        assert(experiment['rho'] is None)
    if experiment['optimizer'] == 'rms':
        assert(experiment['lr'] is not None)
        assert(experiment['lambda'] is None)
        assert(experiment['momentum'] is None)
        assert(experiment['beta_1'] is None)
        assert(experiment['beta_2'] is None)
        assert(experiment['rho'] is not None)

# args ok; start experiment
experiment['start_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

from envs import AcrobotPrediction
from tools import *
import copy
import json
import numpy as np
import torch

# setup libraries
torch.set_num_threads(1)
if experiment['network_seed'] is not None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load test states
test_data = np.load('test_states.npz')
for i in range(len(test_data['y'])):
    test_data['x'][i] = scale_observation(test_data['x'][i])
test_data_x = torch.tensor(test_data['x'], dtype=torch.float)
test_data_y = torch.tensor(test_data['y'], dtype=torch.float)

# load interference test states if necessary
if experiment['approximator'] == 'neural_network':
    interference_test_data = np.load('interference_test_states.npz')
    for i in range(len(interference_test_data['y'])):
        interference_test_data['x'][i] = scale_observation(interference_test_data['x'][i])
        interference_test_data['next_x'][i] = scale_observation(interference_test_data['next_x'][i])
    interference_x_test = torch.tensor(interference_test_data['x'], dtype=torch.float)
    interference_y_test = torch.tensor(interference_test_data['y'], dtype=torch.float)
    interference_next_x_test = torch.tensor(interference_test_data['next_x'], dtype=torch.float)
    interference_next_y_test = torch.tensor(interference_test_data['next_y'], dtype=torch.float)

# prepare buffers to store results
experiment['steps'] = list()
experiment['accuracy'] = list()
if experiment['approximator'] == 'constant':
    experiment['pairwise_interference'] = None
    experiment['activation_overlap'] = None
    experiment['sparse_activation_overlap'] = None
else:
    assert(experiment['approximator'] == 'neural_network')
    experiment['pairwise_interference'] = list()
    experiment['activation_overlap'] = list()
    experiment['sparse_activation_overlap'] = list()

# prepare environment
if experiment['env_seed'] is None:
    env = AcrobotPrediction()
else:
    env = AcrobotPrediction(
        generator=np.random.RandomState(experiment['env_seed']))

# setup predictor
if experiment['approximator'] == 'constant':
    mean = 0
    count = 0
else:
    assert(experiment['approximator'] == 'neural_network')

    # setup network
    linear1 = torch.nn.Linear(6, 100)
    relu1 = torch.nn.ReLU()
    linear2 = torch.nn.Linear(100, 1)
    if experiment['network_seed'] is not None:
        torch.manual_seed(experiment['network_seed'])
    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.normal_(linear1.bias, 0.0, 0.1)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.normal_(linear2.bias, 0.0, 0.1)
    model = torch.nn.Sequential(
        linear1,
        relu1,
        linear2
    )
    if (experiment['loss'] == 'TD') and (experiment['target_update'] > 1):
        target_model = copy.deepcopy(model)
        target_update_counter = 0
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # setup optimizer
    if experiment['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=experiment['lr'],
            momentum=experiment['momentum'])
    elif experiment['optimizer'] == 'rms':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=experiment['lr'],
            alpha=experiment['rho'])
    else:
        assert(experiment['optimizer'] == 'adam')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=experiment['lr'],
            betas=(experiment['beta_1'], experiment['beta_2']))

# build helper functions to compute test statistics
if experiment['approximator'] == 'constant':
    def test():
        with torch.no_grad():
            return ((test_data_y - mean) ** 2).mean().sqrt().item()
else:
    assert(experiment['approximator'] == 'neural_network')

    def test():
        with torch.no_grad():
            return ((test_data_y - model(test_data_x).squeeze()) ** 2).mean().sqrt().item()

    def test_pairwise_interference():
        grads = list()
        for i in range(len(interference_x_test)):
            with torch.no_grad():
                return_ = interference_y_test[i]
                next_return = interference_next_y_test[i]
                if experiment['loss'] == 'squared_error':
                    y = return_
                else:
                    assert(experiment['loss'] == 'TD')
                    y = next_return - return_ + model(interference_next_x_test[i, :])
            y_pred = model(interference_x_test[i, :])
            loss = loss_fn(y_pred, y)
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
        with torch.no_grad():
            activations = list()
            for i in range(len(interference_x_test)):
                activations.append(
                    (relu1.forward(linear1.forward(interference_x_test[i, :]))).numpy())
            mean, count = 0, 0
            for i in range(len(activations)):
                for j in range(i, len(activations)):
                    value = np.mean(np.minimum(activations[i], activations[j]))
                    count += 1
                    mean += (value - mean) / count
        return mean

    def test_sparse_activation_overlap():
        with torch.no_grad():
            activations = list()
            for i in range(len(interference_x_test)):
                activations.append(
                    (linear1.forward(nterference_x_test[i, :]) > 0).numpy())
            mean, count = 0, 0
            for i in range(len(activations)):
                for j in range(i, len(activations)):
                    value = np.mean(np.logical_and(activations[i], activations[j]))
                    count += 1
                    mean += (value - mean) / count
        return mean

# record initial accuracy and interference
experiment['accuracy'].append(test())
if experiment['approximator'] == 'neural_network':
    experiment['pairwise_interference'].append(test_pairwise_interference())
    experiment['activation_overlap'].append(test_activation_overlap())
    experiment['sparse_activation_overlap'].append(test_sparse_activation_overlap())

# run experiment
for episode in range(experiment['num_episodes']):
    observation = env.reset()
    transitions = list()
    done = False
    step = 0
    while not done:
        next_observation, reward, done = env.step()
        step += 1
        transitions.append((observation, reward, next_observation))
        observation = next_observation
    experiment['steps'].append(step)

    # calculate returns
    returns = np.zeros(len(transitions))
    for i, (observation, reward, next_observation) in enumerate(transitions):
        returns[:i + 1] += reward

    # train predictor online on episode
    for i, (observation, reward, next_observation) in enumerate(transitions):

        # ensure consistincy among transitions and returns
        assert(all(abs(next_observation - AcrobotPrediction.get_next_observation(observation)) < 1e-9))
        try:
            assert(all(transitions[i - 1][2] == observation)
                   or (AcrobotPrediction.is_terminal(transitions[i - 1][2])))
        except IndexError:
            pass
        try:
            assert(all(transitions[i + 1][0] == next_observation)
                   or AcrobotPrediction.is_terminal(next_observation))
        except IndexError:
            pass
        assert(returns[i] == AcrobotPrediction.get_return(observation))

        # unpack transition
        if experiment['approximator'] == 'neural_network':
            observation = scale_observation(observation)
            next_observation = scale_observation(next_observation)
        terminal = AcrobotPrediction.is_terminal(next_observation)
        return_ = returns[i]
        try:
            next_return = returns[i + 1]
        except IndexError:
            next_return = 0

        # update predictor
        if experiment['approximator'] == 'constant':
            count += 1
            mean += (return_ - mean) / count
        else:
            assert(experiment['approximator'] == 'neural_network')
            y_pred = model(torch.tensor(observation, dtype=torch.float))
            if experiment['loss'] == 'squared_error':
                y = torch.tensor([return_],
                                 dtype=torch.float)
            else:
                assert(experiment['loss'] == 'TD')
                if terminal:
                    y = torch.tensor([- 1],
                                     dtype=torch.float)
                else:
                    with torch.no_grad():  # don't use residual gradient
                        if experiment['target_update'] > 1:
                            y = reward + target_model(torch.tensor(next_observation,
                                                                   dtype=torch.float))
                        else:
                            y = reward + model(torch.tensor(next_observation,
                                                            dtype=torch.float))
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (experiment['loss'] == 'TD') and (experiment['target_update'] > 1):
                target_update_counter += 1
                assert(target_update_counter <= experiment['target_update'])
                if target_update_counter == experiment['target_update']:
                    target_model.load_state_dict(model.state_dict())
                    target_update_counter = 0

    # record test statistics
    experiment['accuracy'].append(test())
    if experiment['approximator'] == 'neural_network':
        experiment['pairwise_interference'].append(test_pairwise_interference())
        experiment['activation_overlap'].append(test_activation_overlap())
        experiment['sparse_activation_overlap'].append(test_sparse_activation_overlap())

# save results
experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
if experiment['outfile'] == '-':
    print(json.dumps(experiment, sort_keys=True))
else:
    assert(not os.path.isfile(experiment['outfile']))
    with open(experiment['outfile'], 'w') as outfile:
        json.dump(experiment, outfile, sort_keys=True)
