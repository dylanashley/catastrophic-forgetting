# -*- coding: utf-8 -*-

import collections
import json
import numpy as np
import pandas as pd

# prevent recursive imports
GOAL_POSITION = 0.5
OBSERVATION_MAX = np.array([0.6, 0.07])
OBSERVATION_MIN = np.array([- 1.2, - 0.07])

misc_labels = [
    'end_time',
    'env_range',
    'num_episodes',
    'outfile',
    'start_time',
    'steps']
repeat_labels = [
    'env_seed',
    'network_seed']
result_labels = [
    'accuracy',
    'activation_overlap',
    'auc',
    'pairwise_interference']
hyperparameter_labels = [
    'approximator',
    'beta_1',
    'beta_2',
    'lambda_',
    'loss',
    'lr',
    'momentum',
    'optimizer',
    'rho',
    'target_update']

Key = collections.namedtuple('Key', hyperparameter_labels)

def to_nested_tuples(item):
    """Converts lists and nested lists to tuples and nested tuples.

    Returned value should be hashable.
    """
    if isinstance(item, list):
        return tuple([to_nested_tuples(i) for i in item])
    else:
        return item

def list_of_dicts_to_dict_of_lists(list_):
    """Converts a list of dictionaries to a dictionaries of lists.

    Useful when building dataframes.
    """
    rv = dict()
    for item in list_:
        for key, value in item.items():
            if key not in rv:
                rv[key] = list()
            rv[key].append(value)
    return rv

def get_hyperparameter_key(entry):
    """Obtains the hyperparameter settings from an experimental result.

    Checks returned value is hashable.
    """
    rv = list()
    for label in hyperparameter_labels:
        if isinstance(entry[label], float) and np.isnan(entry[label]):
            rv.append(None)
        else:
            rv.append(entry[label])
        hash(rv[-1])
    return Key(* rv)

def scale(value, start_min, start_max, end_min, end_max):
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return (end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min))

def scale_observation(value):
    assert(all([OBSERVATION_MIN[i] <= value[i] <= OBSERVATION_MAX[i] for i in range(len(value))]))
    rv = scale(value, OBSERVATION_MIN, OBSERVATION_MAX, - 1, 1)
    assert(all([- 1 <= rv[i] <= 1 for i in range(len(rv))]))
    return rv
