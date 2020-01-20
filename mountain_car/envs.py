# -*- coding: utf-8 -*-

from tools import *
import numpy as np


class MountainCar:

    EPS = 1e-9
    GOAL_POSITION = 0.5
    MAX_POSITION = 0.6
    MAX_VELOCITY = 0.07
    MIN_POSITION = - 1.2
    MIN_VELOCITY = - MAX_VELOCITY

    def __init__(self, generator=None):
        self.position = 0.0
        self.velocity = 0.0
        if generator is None:
            self.generator = np.random.RandomState()
        else:
            self.generator = generator

    def reset(self):
        self.position = scale(self.generator.rand(), 0, 1,
                              self.MIN_POSITION, self.GOAL_POSITION)
        self.velocity = scale(self.generator.rand(), 0, 1,
                              self.MIN_VELOCITY, self.MAX_VELOCITY)
        assert(self.position < self.GOAL_POSITION)
        return (self.position, self.velocity)

    def set_position(self, position):
        assert(self.MIN_POSITION <= position <= self.MAX_POSITION)
        self.position = position

    def set_velocity(self, velocity):
        assert(self.MIN_VELOCITY <= velocity <= self.MAX_VELOCITY)
        self.velocity = velocity

    def step(self, action):
        self.position, self.velocity = self.get_next_observation((self.position, self.velocity),
                                                                 action)
        done = self.position >= self.GOAL_POSITION
        reward = 0 if done else - 1
        return (self.position, self.velocity), reward, done

    @staticmethod
    def get_next_observation(observation, action):
        position = observation[0]
        velocity = observation[1]
        velocity += (action - 1) * 0.001 + np.cos(3 * position) * (- 0.0025)
        velocity = np.clip(velocity, MountainCar.MIN_VELOCITY, MountainCar.MAX_VELOCITY)
        position += velocity
        position = np.clip(position, MountainCar.MIN_POSITION, MountainCar.MAX_POSITION)
        if (abs(position - MountainCar.MIN_POSITION) < MountainCar.EPS) and (velocity < 0):
            velocity = 0.0
        return (position, velocity)


class MountainCarPrediction:

    def __init__(self, generator=None):
        self._env = MountainCar(generator=generator)

    def reset(self):
        return self._env.reset()

    def set_position(self, position):
        self._env.set_position(position)

    def set_velocity(self, velocity):
        self._env.set_velocity(velocity)

    def step(self):
        return self._env.step(self.get_next_action((self._env.position, self._env.velocity)))

    @staticmethod
    def get_next_action(observation):
        return 0 if observation[1] < 0 else 2

    @staticmethod
    def get_next_observation(observation):
        return MountainCar.get_next_observation(observation,
                                                MountainCarPrediction.get_next_action(observation))

    @staticmethod
    def get_return(observation):
        env = MountainCarPrediction()
        env.reset()
        env.set_position(observation[0])
        env.set_velocity(observation[1])
        done = False
        rv = 0
        while not done:
            _, reward, done = env.step()
            rv += reward
        return rv
