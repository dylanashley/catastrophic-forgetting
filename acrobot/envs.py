# -*- coding: utf-8 -*-

import gym
import numpy as np


class AcrobotPrediction:

    def __init__(self, generator=None):
        if generator is None:
            self.generator = np.random.RandomState()
        else:
            self.generator = generator
        self._env = gym.make('Acrobot-v1')
        self._env._max_episode_steps = float('inf')
        self._env.seed(self.generator.randint(2 ** 16))
        self._env.observation_space.seed(self.generator.randint(2 ** 16))

    def get_next_observation(self, observation):
        old_observation = self._env.env._get_ob()
        self.set_state(observation)
        observation, _, _ = self.step()
        self.set_state(old_observation)
        return observation

    def get_random_observation(self):
        return self._env.observation_space.sample()

    def reset(self):
        return self._env.reset()

    def set_state(self, observation):
        self._env.env.state[0] = (- 1 if observation[1] < 0 else 1) * np.arccos(observation[0])
        self._env.env.state[1] = (- 1 if observation[3] < 0 else 1) * np.arccos(observation[2])
        self._env.env.state[2] = observation[4]
        self._env.env.state[3] = observation[5]

    def step(self):
        observation, reward, done, _ = self._env.step(
            self.get_next_action(self._env.env._get_ob()))
        return observation, reward, done

    @staticmethod
    def get_next_action(observation):
        return 2 if observation[4] < 0 else 0

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
