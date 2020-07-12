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
         'defaults to 1',
    default=1)
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
test_data = dict(np.load('test_states.npz'))
for i in range(len(test_data['y'])):
    test_data['x'][i] = scale_observation(test_data['x'][i])
test_data_x = torch.tensor(test_data['x'], dtype=torch.float)
test_data_y = torch.tensor(test_data['y'], dtype=torch.float)

# load interference test states if necessary
if experiment['approximator'] == 'neural_network':
    interference_test_data = dict(np.load('interference_test_states.npz'))
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
    experiment['activation_similarity'] = None
    experiment['pairwise_interference'] = None
else:
    assert(experiment['approximator'] == 'neural_network')
    experiment['activation_similarity'] = list()
    experiment['pairwise_interference'] = list()

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
    linear1 = torch.nn.Linear(6, 32)
    relu1 = torch.nn.ReLU()
    linear2 = torch.nn.Linear(32, 256)
    relu2 = torch.nn.ReLU()
    linear3 = torch.nn.Linear(256, 1)
    if experiment['network_seed'] is not None:
        torch.manual_seed(experiment['network_seed'])
    torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity='relu')
    torch.nn.init.normal_(linear1.bias, std=0.1)
    torch.nn.init.kaiming_uniform_(linear2.weight, nonlinearity='relu')
    torch.nn.init.normal_(linear2.bias, std=0.1)
    torch.nn.init.kaiming_uniform_(linear3.weight, nonlinearity='relu')
    torch.nn.init.normal_(linear3.bias, std=0.1)
    model = torch.nn.Sequential(
        linear1,
        relu1,
        linear2,
        relu2,
        linear3
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
    @torch.no_grad()
    def test():
        return ((test_data_y - mean) ** 2).mean().sqrt().item()
else:
    assert(experiment['approximator'] == 'neural_network')

    @torch.no_grad()
    def test():
        return float(((test_data_y - model(test_data_x).squeeze()) ** 2).mean().sqrt().item())

    @torch.no_grad()
    def test_activation_similarity():
        activations = list()
        for i in range(len(interference_x_test)):
            layer1 = relu1.forward(linear1.forward(interference_x_test[i, :]))
            layer2 = relu2.forward(layer1)
            activations.append(np.concatenate((layer1.numpy(), layer2.numpy())))
        mean, count = 0, 0
        for i in range(len(activations)):
            for j in range(i, len(activations)):
                value = np.dot(activations[i], activations[j])
                count += 1
                mean += (value - mean) / count
        return float(mean)

    @torch.no_grad()
    def _interference_test():
        return ((interference_y_test - model(interference_x_test).squeeze()) ** 2).numpy()

    def test_pairwise_interference():
        pre_performance = np.tile(_interference_test(), (len(interference_x_test), 1))
        post_performance = np.zeros_like(pre_performance)
        state_dict = copy.deepcopy(model.state_dict())
        optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        for i in range(len(interference_x_test)):
            y_pred = model(interference_x_test[i, :])
            y = interference_y_test[i]
            loss = loss_fn(y_pred, y.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            post_performance[i, :] = _interference_test()
            model.load_state_dict(copy.deepcopy(state_dict))
            optimizer.load_state_dict(copy.deepcopy(optimizer_state_dict))
        return float(np.mean(post_performance - pre_performance))

# record initial accuracy and interference
experiment['accuracy'].append(test())
if experiment['approximator'] == 'neural_network':
    experiment['activation_similarity'].append(test_activation_similarity())
    experiment['pairwise_interference'].append(test_pairwise_interference())

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
    returns = np.arange(- len(transitions), 1)

    # train predictor online on episode
    for i, (observation, reward, next_observation) in enumerate(transitions):

        # ensure consistincy among transitions and returns
        if i > 0:
            assert(transitions[i - 1][2] == observation)
        if i < len(transitions) - 1:
            assert(transitions[i + 1][0] == next_observation)

        # update predictor
        if experiment['approximator'] == 'constant':
            count += 1
            mean += (returns[i] - mean) / count
        else:
            assert(experiment['approximator'] == 'neural_network')
            y_pred = model(torch.tensor(scale_observation(observation), dtype=torch.float))
            if experiment['loss'] == 'squared_error':
                y = torch.tensor([returns[i]], dtype=torch.float)
            else:
                assert(experiment['loss'] == 'TD')
                if AcrobotPrediction.is_terminal(next_observation):
                    y = torch.tensor([- 1], dtype=torch.float)
                else:
                    with torch.no_grad():  # don't use residual gradient
                        if experiment['target_update'] > 1:
                            y = reward + \
                                target_model(torch.tensor(scale_observation(next_observation),
                                                          dtype=torch.float))
                        else:
                            y = reward + \
                                model(torch.tensor(scale_observation(next_observation),
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
        experiment['activation_similarity'].append(test_activation_similarity())
        experiment['pairwise_interference'].append(test_pairwise_interference())

# save results
experiment['end_time'] = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
if experiment['outfile'] == '-':
    print(json.dumps(experiment, sort_keys=True))
else:
    assert(not os.path.isfile(experiment['outfile']))
    with open(experiment['outfile'], 'w') as outfile:
        json.dump(experiment, outfile, sort_keys=True)
