# -*- coding: utf-8 -*-

import collections
import json
import numpy as np
import pandas as pd

misc_labels = [
    'end_time',
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
    'beta_1',
    'beta_2',
    'lr',
    'momentum',
    'optimizer',
    'rho']

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
