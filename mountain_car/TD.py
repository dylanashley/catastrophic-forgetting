# -*- coding: ascii -*-

import numpy as np


class TD:

    def __init__(self, initial_x):
        self._last_x = np.copy(initial_x)
        self.e = np.zeros(initial_x.shape)
        self.w = np.zeros(initial_x.shape)

    def __call__(self, x):
        """Return the current prediction for a given set of features x."""
        return np.dot(self.w, x)

    def reset(self, initial_x):
        """Prepare the learner for a new episode."""
        self.update(0, 0, initial_x, 0, 0)

    def update(self, reward, gamma, x, alpha, lambda_):
        delta = reward + gamma * self(x) - self(self._last_x)
        self.w += alpha * delta * self.e
        self.e *= lambda_ * gamma
        self.e += x
        np.copyto(self._last_x, x)
