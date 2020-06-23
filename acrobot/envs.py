#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import os
import sys
import warnings

from tools import *


class Acrobot:

    NUM_ACTIONS = 3
    OBSERVATION_MAX = np.array([1.0, 1.0, 1.0, 1.0, 4 * np.pi, 9 * np.pi])
    OBSERVATION_MIN = - OBSERVATION_MAX

    def __init__(self, generator=None):
        if generator is None:
            self.generator = np.random.RandomState()
        else:
            self.generator = generator
        self._env = gym.make('Acrobot-v1')
        self._env._max_episode_steps = float('inf')
        self._env.seed(self.generator.randint(2 ** 16))
        self._env.observation_space.seed(self.generator.randint(2 ** 16))

    def reset(self):
        return self._env.reset()

    def get_state(self):
        return self._env.env._get_ob()

    def set_state(self, observation):
        for i, v in enumerate(Acrobot._observation_to_state(observation)):
            self._env.state[i] = v

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        return observation, reward, done

    @staticmethod
    def _observation_to_state(observation):
        state = list()
        state.append((- 1 if observation[1] < 0 else 1) * np.arccos(observation[0]))
        state.append((- 1 if observation[3] < 0 else 1) * np.arccos(observation[2]))
        state.append(observation[4])
        state.append(observation[5])
        return tuple(state)

    @staticmethod
    def get_next_observation(observation, action):
        env = Acrobot()
        env.reset()
        env.set_state(observation)
        observation, _, _ = env.step(action)
        return tuple(observation)

    @staticmethod
    def is_terminal(observation):
        state = Acrobot._observation_to_state(observation)
        return bool(- np.cos(state[0]) - np.cos(state[1] + state[0]) > 1)


class AcrobotPrediction:

    NUM_ACTIONS = Acrobot.NUM_ACTIONS
    OBSERVATION_MAX = Acrobot.OBSERVATION_MAX
    OBSERVATION_MIN = Acrobot.OBSERVATION_MIN

    def __init__(self, generator=None):
        self._env = Acrobot(generator=generator)

    def reset(self):
        return self._env.reset()

    def get_state(self):
        return self._env.get_state()

    def set_state(self, observation):
        self._env.set_state(observation)

    def step(self):
        return self._env.step(AcrobotPrediction.get_next_action(self._env.get_state()))

    def get_next_action(observation):
        if abs(observation[5]) > 10 * abs(observation[4]):
            return 0  # prevent windmill phenomenon
        elif observation[4] < 0:
            return - 1
        elif observation[4] > 0:
            return 1
        else:
            return 0

    @staticmethod
    def get_next_observation(observation):
        return Acrobot.get_next_observation(observation,
                                            AcrobotPrediction.get_next_action(observation))

    @staticmethod
    def get_return(observation):
        env = AcrobotPrediction()
        env.reset()
        env.set_state(observation)
        done = False
        rv = 0
        while not done:
            _, reward, done = env.step()
            rv += reward
        return rv

    @staticmethod
    def is_terminal(observation):
        return Acrobot.is_terminal(observation)

if __name__ == '__main__':
    if os.path.isfile('policy_steps.npy'):
        warnings.warn('outfile already exists; terminating\n')
        sys.exit(0)

    SEED = 57009  # generated by RANDOM.org

    env = AcrobotPrediction(np.random.RandomState(SEED))
    steps = list()
    try:
        for episode in range(1000000):
            observation = env.reset()
            done = False
            step = 0
            while not done:
                observation, _, done = env.step()
                step += 1
            steps.append(step)
        np.save('policy_steps.npy', np.array(steps, dtype=int))
    except KeyboardInterrupt:
        steps.append(step)
    finally:
        print('COUNT: {0}'.format(len(steps)))
        print(' MEAN: {0:.4f}'.format(np.mean(steps)))
        print('  STD: {0:.4f}'.format(np.std(steps)))
        print('  SEM: {0:.4f}'.format(np.std(steps) / len(steps)))
        print('  MIN: {0}'.format(np.min(steps)))
        print('  MAX: {0}'.format(np.max(steps)))
