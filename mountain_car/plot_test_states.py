#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCar
from tools import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load the data
temp = np.load('test_states.npz')
x = temp['x']
y = temp['y']
z = np.load('test_states_trajectory.npy')

# create the plot
MIN_POSITION = MountainCar.MIN_POSITION
MAX_POSITION = MountainCar.MAX_POSITION
MIN_VELOCITY = MountainCar.MIN_VELOCITY
MAX_VELOCITY = MountainCar.MAX_VELOCITY
position_bins = np.linspace(MIN_POSITION, MAX_POSITION, 25, endpoint=True)
velocity_bins = np.linspace(MIN_VELOCITY, MAX_VELOCITY, 25, endpoint=True)
xticks = np.linspace(MIN_POSITION, MAX_POSITION, 5, endpoint=True)
yticks = np.linspace(MIN_VELOCITY, MAX_VELOCITY, 5, endpoint=True)
sns.set_style('ticks')
fig, axmat = plt.subplots(1, 2, dpi=300, figsize=(12, 4))
x_counts, _, _, x_map = axmat[0].hist2d(
    x[:, 0],
    x[:, 1],
    bins=[position_bins, velocity_bins],
    vmin=0,
    cmap='gray_r')
z_counts, _, _, z_map = axmat[1].hist2d(
    z[:, 0],
    z[:, 1],
    bins=[position_bins, velocity_bins],
    vmin=0,
    cmap='gray_r')
for ax in axmat:
    ax.set_xlabel('Position', labelpad=5)
    ax.set_ylabel('Velocity', labelpad=5)
axmat[0].set_title('Test States', pad=10)
axmat[1].set_title('Test States Trajectory', pad=10)
plt.colorbar(x_map, ax=axmat[0])
plt.colorbar(z_map, ax=axmat[1])
fig.subplots_adjust(wspace=0.4)

# save the plot
fig.savefig('test_states.png', bbox_inches='tight')
