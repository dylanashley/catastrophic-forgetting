#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCar, MountainCarPrediction
from tools import scale
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
sns.set_style('ticks')
fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
sns.heatmap(return_grid,
            vmax=0,
            cmap='cividis',
            square=True,
            ax=ax)
for _, spine in ax.spines.items():
    spine.set_visible(True)
fig.axes[1].artists[0]._linewidth = list(ax.spines.values())[0]._linewidth
ax.set_xlabel('Position', labelpad=5)
plt.xticks(scale(np.array([- 1.0, - 0.5, 0.0, 0.5]),
                 MountainCar.MIN_POSITION,
                 MountainCar.MAX_POSITION,
                 0,
                 24),
           labels=[- 1.0, - 0.5, 0.0, 0.5],
           rotation='horizontal')
ax.set_ylabel('Velocity', labelpad=5)
plt.yticks(scale(np.array([- 0.06, - 0.03, 0.0, 0.03, 0.06]),
                 MountainCar.MIN_VELOCITY,
                 MountainCar.MAX_VELOCITY,
                 0,
                 24),
           labels=[- 0.06, - 0.03, 0.0, 0.03, 0.06],
           rotation='horizontal')
fig.savefig('state_values.pdf', bbox_inches='tight')
