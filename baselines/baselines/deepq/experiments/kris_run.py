#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main(args):

    # setup optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=float(args['lr']),
        beta1=float(args['beta_1']),
        beta2=float(args['beta_2']))

    # setup env
    env = gym.make('MountainCar-v0')
    env.seed(args['seed'])

    # train network
    act = deepq.learn(
        env,
        network=models.mlp(num_hidden=64, num_layers=1),
        optimizer=optimizer,
        total_timesteps=100000,
        train_freq=int(args['train_freq']),
        buffer_size=int(args['buffer_size']),
        exploration_fraction=float(args['exploration_fraction']),
        exploration_final_eps=0.1,
        print_freq=None,
        target_network_update_freq=int(args['target_network_update_freq']))

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
    print('seed,lr,beta_1,beta_2,buffer_size,train_freq,target_network_update_freq,exploration_fraction,mean_return')
    print('{0},{1},{2},{3},{4},{5},{6},{7},{8:.6f}'.format(
        args['seed'],
        args['lr'],
        args['beta_1'],
        args['beta_2'],
        args['buffer_size'],
        args['train_freq'],
        args['target_network_update_freq'],
        args['exploration_fraction'],
        mean_return))


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is the Gym experiment proposed by Kris.')
    parser.add_argument(
        'seed',
        type=int,
        help='random seed for environment')
    parser.add_argument(
        'lr',
        type=str,
        help='learning rate for training')
    parser.add_argument(
        'beta_1',
        type=str,
        help='beta 1 hyperparameter for adam')
    parser.add_argument(
        'beta_2',
        type=str,
        help='beta 2 hyperparameter for adam')
    parser.add_argument(
        'buffer_size',
        type=int,
        help='experience replay buffer size')
    parser.add_argument(
        'train_freq',
        type=int,
        help='number of steps between updates to the model')
    parser.add_argument(
        'target_network_update_freq',
        type=int,
        help='number of steps between updates to the target network')
    parser.add_argument(
        'exploration_fraction',
        type=str,
        help='epsilon decay rate for exploration')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
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
