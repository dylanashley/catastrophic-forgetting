#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCar, MountainCarPrediction
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# get grid of state values
position_bins = np.linspace(MountainCar.MIN_POSITION,
                            MountainCar.MAX_POSITION,
                            25,
                            endpoint=False)
velocity_bins = np.linspace(MountainCar.MIN_VELOCITY,
                            MountainCar.MAX_VELOCITY,
                            25,
                            endpoint=False)
position_bins += (position_bins[1] - position_bins[0]) / 2
velocity_bins += (velocity_bins[1] - velocity_bins[0]) / 2
return_grid = np.ones((len(velocity_bins), len(position_bins))) * np.nan
for i, velocity in enumerate(velocity_bins):
    for j, position in enumerate(position_bins):
        return_grid[i, j] = \
            MountainCarPrediction.get_return((position, velocity))

# create the plot
fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
sns.heatmap(return_grid, cmap='viridis', square=True, yticklabels=False, xticklabels=False, ax=ax)
ax.set_xlabel('Position', labelpad=5)
ax.set_ylabel('Velocity', labelpad=5)
ax.set_title('Value of States in Mountain Car Prediction Task', fontsize=10)
fig.savefig('state_values.png')
