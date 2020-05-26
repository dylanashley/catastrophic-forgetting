#!/usr/bin/env python
# -*- coding: utf-8 -*-

from envs import MountainCar
from tools import *
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# parse args
parser = argparse.ArgumentParser(
    description='This plots the distribution of states for a mountain car prediction task.')
parser.add_argument(
    'infile',
    type=str,
    help='npz file to load test states from')
parser.add_argument(
    'trajectory_infile',
    type=str,
    help='npy file to load test states trajectory from')
parser.add_argument(
    'outfile',
    type=str,
    help='file to save plot to')
args = vars(parser.parse_args())

# load the data
temp = np.load(args['infile'])
x = temp['x']
y = temp['y']
z = np.load(args['trajectory_infile'])

# create the plot
position_bins = np.linspace(MountainCar.MIN_POSITION,
                            MountainCar.MAX_POSITION,
                            25,
                            endpoint=False)
velocity_bins = np.linspace(MountainCar.MIN_VELOCITY,
                            MountainCar.MAX_VELOCITY,
                            25,
                            endpoint=False)
sns.set_style('ticks')
fig, axmat = plt.subplots(1, 2, dpi=300, figsize=(12, 4))
x_counts, _, _, x_map = axmat[0].hist2d(
    x[:, 0],
    x[:, 1],
    bins=[position_bins, velocity_bins],
    vmin=0,
    cmap='cividis')
z_counts, _, _, z_map = axmat[1].hist2d(
    z[:, 0],
    z[:, 1],
    bins=[position_bins, velocity_bins],
    vmin=0,
    cmap='cividis')
for ax in axmat:
    ax.set_xlabel('Position', labelpad=5)
    ax.set_xticks([- 1.0, - 0.5, 0.0, 0.5])
    ax.set_ylabel('Velocity', labelpad=5)
    ax.set_yticks([- 0.06, - 0.03, 0.0, 0.03, 0.06])
axmat[0].set_title('Sample', pad=10)
axmat[1].set_title('Trajectory', pad=10)
plt.colorbar(x_map, ax=axmat[0])
plt.colorbar(z_map, ax=axmat[1])
fig.subplots_adjust(wspace=0.4)

# save the plot
fig.savefig(args['outfile'], bbox_inches='tight')
