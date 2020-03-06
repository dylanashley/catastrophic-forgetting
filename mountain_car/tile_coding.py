# -*- coding: ascii -*-

from envs import MountainCar

NUMBER_OF_TILINGS = 8
TILING_CARDINALITY = 10
POSITION_RANGE = (MountainCar.MIN_POSITION, MountainCar.MAX_POSITION)
VELOCITY_RANGE = (MountainCar.MIN_VELOCITY, MountainCar.MAX_VELOCITY)

TILING_AREA = TILING_CARDINALITY**2
TILE_COUNT = TILING_AREA * NUMBER_OF_TILINGS
POSITION_RANGE_SIZE = float(max(POSITION_RANGE) - min(POSITION_RANGE))
VELOCITY_RANGE_SIZE = float(max(VELOCITY_RANGE) - min(VELOCITY_RANGE))


def discretize(position, velocity):
    indices = list()
    position = (position - min(POSITION_RANGE)) / POSITION_RANGE_SIZE
    velocity = (velocity - min(VELOCITY_RANGE)) / VELOCITY_RANGE_SIZE
    for tiling in range(NUMBER_OF_TILINGS):

        offset = 0 if NUMBER_OF_TILINGS == 1 else \
            tiling / float(NUMBER_OF_TILINGS)

        position_index = int(position * (TILING_CARDINALITY - 1) + offset)
        position_index = min(position_index, TILING_CARDINALITY - 1)

        velocity_index = int(velocity * (TILING_CARDINALITY - 1) + offset)
        velocity_index = min(velocity_index, TILING_CARDINALITY - 1)

        indices.append(position_index
                       + velocity_index * TILING_CARDINALITY
                       + TILING_AREA * tiling)
    return indices
