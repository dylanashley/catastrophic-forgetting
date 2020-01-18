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

    def step(self, action):
        self.velocity += (action - 1) * 0.001 + np.cos(3 * self.position) * (- 0.0025)
        self.velocity = np.clip(self.velocity, self.MIN_VELOCITY, self.MAX_VELOCITY)
        self.position += self.velocity
        self.position = np.clip(self.position, self.MIN_POSITION, self.MAX_POSITION)
        if (abs(self.position - self.MIN_POSITION) < self.EPS) and (self.velocity < 0):
            self.velocity = 0.0
        done = self.position >= self.GOAL_POSITION
        reward = 0 if done else - 1
        return (self.position, self.velocity), reward, done

    def set_position(self, position):
        assert(self.MIN_POSITION <= position <= self.MAX_POSITION)
        self.position = position

    def set_velocity(self, velocity):
        assert(self.MIN_VELOCITY <= velocity <= self.MAX_VELOCITY)
        self.velocity = velocity


class MountainCarPrediction:

    def __init__(self, generator=None):
        self._env = MountainCar(generator=generator)

    def reset(self):
        return self._env.reset()

    def step(self):
        return self._env.step(0 if self._env.velocity < 0 else 2)

    def set_position(self, position):
        self._env.set_position(position)

    def set_velocity(self, velocity):
        self._env.set_velocity(velocity)

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
