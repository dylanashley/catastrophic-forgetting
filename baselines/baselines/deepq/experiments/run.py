#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main(args):

    # setup optimizer
    if args['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=float(args['lr']),
            beta1=float(args['beta_1']),
            beta2=float(args['beta_2']))
    else:
        assert args['optimizer'] == 'sgd'
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=float(args['lr']))

    # setup env
    if args['env'] == 'mountain_car':
        env = gym.make('MountainCar-v0')
    else:
        assert args['env'] == 'cartpole'
        env = gym.make('CartPole-v0')
    env.seed(args['seed'])

    # train network
    if args['env'] == 'mountain_car':
        act = deepq.learn(
            env,
            network=models.mlp(num_hidden=64, num_layers=1),
            optimizer=optimizer,
            total_timesteps=100000,
            buffer_size=int(args['buffer_size']),
            exploration_fraction=float(args['exploration_fraction']),
            exploration_final_eps=0.1,
            print_freq=None)
    else:
        assert args['env'] == 'cartpole'
        act = deepq.learn(
            env,
            network='mlp',
            optimizer=optimizer,
            total_timesteps=100000,
            buffer_size=int(args['buffer_size']),
            exploration_fraction=float(args['exploration_fraction']),
            exploration_final_eps=0.02,
            print_freq=None,
            callback=lambda lcl, _glb:
                lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199)

    # evaluate model
    mean_return, return_count = 0, 0
    for _ in range(1000):
        obs, done = env.reset(), False
        return_ = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            return_ += rew
        return_count += 1
        mean_return += (return_ - mean_return) / return_count
    print('seed,env,optimizer,lr,buffer_size,exploration_fraction,beta_1,beta_2,mean_return')
    print('{},{},{},{},{},{},{},{},{}'.format(
        args['seed'],
        args['env'],
        args['optimizer'],
        args['lr'],
        args['buffer_size'],
        args['exploration_fraction'],
        args['beta_1'],
        args['beta_2'],
        mean_return))


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is the main Gym experiment.')
    parser.add_argument(
        'env',
        choices=['mountain_car', 'cartpole'],
        help='environment to run')
    parser.add_argument(
        'seed',
        type=int,
        help='random seed for environment')
    subparsers = parser.add_subparsers(
        dest='optimizer',
        help='optimizer for training on both datasets')
    sgd_parser = subparsers.add_parser('sgd')
    adam_parser = subparsers.add_parser('adam')
    for subparser in [sgd_parser, adam_parser]:
        subparser.add_argument(
            'lr',
            type=str,
            help='learning rate for training on both datasets')
        subparser.add_argument(
            'buffer_size',
            type=int,
            help='experience replay buffer size')
        subparser.add_argument(
            'exploration_fraction',
            type=str,
            help='epsilon decay rate for exploration')
    adam_parser.add_argument(
        'beta_1',
        type=str,
        help='beta 1 hyperparameter for adam on both datasets')
    adam_parser.add_argument(
        'beta_2',
        type=str,
        help='beta 2 hyperparameter for adam on both datasets')
    args = vars(parser.parse_args())
    if 'beta_1' not in args:
        assert('beta_2' not in args)
        args['beta_1'] = None
        args['beta_2'] = None
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import tensorflow as tf
    tf.logging.set_verbosity('FATAL')

    import warnings
    warnings.filterwarnings('ignore')

    import argparse
    import gym

    from baselines import deepq
    from baselines.common import models

    args = parse_args()
    main(args)
