#!/usr/bin/env python


def main(args):
    # for lr in ['1e-2', '1e-3', '1e-4']:
    #     for exploration_fraction in ['0.01', '0.1', '0.25']:
    lr = args['lr']
    exploration_fraction = args['exploration_fraction']

    # setup optimizer
    if args['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=float(args['lr']))
    else:
        assert args['optimizer'] == 'gradient_descent'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=float(args['lr']))

    # setup env
    if args['env'] == 'mountain_car':
        env = gym.make('MountainCar-v0')
    else:
        assert args['env'] == 'cartpole'
        env = gym.make('CartPole-v0')

    # train network
    if args['env'] == 'mountain_car':
        act = deepq.learn(
            env,
            network=models.mlp(num_hidden=64, num_layers=1),
            optimizer=optimizer,
            total_timesteps=100000,
            buffer_size=1,
            batch_size=1,
            learning_starts=1,
            target_network_update_freq=500,
            exploration_fraction=float(exploration_fraction),
            exploration_final_eps=0.1,
            print_freq=None
        )
    else:
        assert args['env'] == 'cartpole'
        act = deepq.learn(
            env,
            network='mlp',
            optimizer=optimizer,
            total_timesteps=100000,
            buffer_size=50000,
            exploration_fraction=args['exploration_fraction'],
            exploration_final_eps=0.02,
            print_freq=None,
            callback=lambda lcl, _glb: lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
        )

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
    print(mean_return)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', choices=['mountain_car', 'cartpole'])
    parser.add_argument('optimizer', choices=['adam', 'gradient_descent'])
    parser.add_argument('lr', type=str)
    parser.add_argument('buffer_size', type=int)
    parser.add_argument('exploration_fraction', type=float)
    return vars(parser.parse_args())

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
