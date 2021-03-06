#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCar, MountainCarPrediction
import argparse
import numpy as np
import os
import warnings

# parse args
parser = argparse.ArgumentParser(
    description='This constructs a test set for a mountain car prediction task.')
parser.add_argument(
    'outfile',
    type=str,
    help='npz file to dump states and returns; '
         'will terminate if file already exists')
parser.add_argument(
    'env_range',
    choices=['full', 'classic'],
    help='range to sample initial position and velocity values for new episodes')
parser.add_argument(
    'num_steps',
    type=int,
    help='length of trajectory to sample states from')
parser.add_argument(
    'sample_size',
    type=int,
    help='number of states to sample from trajectory')
parser.add_argument(
    '--trajectory-outfile',
    type=str,
    help='npy file to dump the trajectory the sample is taken from to; '
         'will terminate if this argument is specified but file already exists')
parser.add_argument(
    '--interference-outfile',
    type=str,
    help='npz file to dump states and returns for use in measuring catastrophic interference; '
         'will terminate if this argument is specified but file already exists')
args = vars(parser.parse_args())

# check args
assert(0 < args['sample_size'] <= args['num_steps'])
if os.path.isfile(args['outfile']):
    warnings.warn('outfile already exists; terminating\n')
    sys.exit(0)
if args['trajectory_outfile'] is not None:
    if os.path.isfile(args['trajectory_outfile']):
        warnings.warn('trajectory outfile already exists; terminating\n')
        sys.exit(0)
if args['interference_outfile'] is not None:
    if os.path.isfile(args['interference_outfile']):
        warnings.warn('interference outfile already exists; terminating\n')
        sys.exit(0)

SEED = 49192  # generated by RANDOM.ORG

# get trajectory
states = list()
generator = np.random.RandomState(SEED)
env = MountainCarPrediction(generator=generator)
done = True
while len(states) < args['num_steps']:
    if done:
        state = env.reset(range=args['env_range'])
        done = False
    else:
        state, reward, done = env.step()
    if not done:
        states.append(state)

# sample states from the trajectory
indices = generator.choice(len(states), args['sample_size'])
sample = [states[i] for i in indices]

# get corresponding returns for states in the sample
returns = [MountainCarPrediction.get_return(i) for i in sample]

# get interference interference states if requested
if args['interference_outfile'] is not None:
    position_bins = np.linspace(MountainCar.MIN_POSITION,
                                MountainCar.GOAL_POSITION,
                                6,
                                endpoint=False)
    velocity_bins = np.linspace(MountainCar.MIN_VELOCITY,
                                MountainCar.MAX_VELOCITY,
                                6,
                                endpoint=False)
    position_bins += (position_bins[1] - position_bins[0]) / 2
    velocity_bins += (velocity_bins[1] - velocity_bins[0]) / 2
    interference_states = list()
    interference_returns = list()
    interference_next_states = list()
    interference_next_returns = list()
    for position in position_bins:
        for velocity in velocity_bins:
            observation = (position, velocity)
            interference_states.append(observation)
            interference_returns.append(MountainCarPrediction.get_return(observation))
            next_observation = MountainCarPrediction.get_next_observation(observation)
            interference_next_states.append(next_observation)
            interference_next_returns.append(MountainCarPrediction.get_return(next_observation))

# save arrays
np.savez(args['outfile'],
         x=sample,
         y=returns)
if args['trajectory_outfile'] is not None:
    np.save(args['trajectory_outfile'],
            states)
if args['interference_outfile'] is not None:
    np.savez(args['interference_outfile'],
             x=interference_states,
             y=interference_returns,
             next_x=interference_next_states,
             next_y=interference_next_returns)
