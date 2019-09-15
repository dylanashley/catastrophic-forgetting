# -*- coding: utf-8 -*-

import collections
import json

from typing import Any, Dict, Hashable, List, NamedTuple

setting_labels = [
    'architecture',
    'beta_1',
    'beta_2',
    'criteria',
    'dataset',
    'digits',
    'folds',
    'has_all_digits_phase',
    'lr',
    'momentum',
    'optimizer',
    'phases',
    'rho',
    'tolerance']
repeat_labels = [
    'seed',
    'test_fold',
    'validation_fold']
result_labels = [
    'accuracies',
    'outfile',
    'phase_length',
    'predictions',
    'success']

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

def get_setting_key(entry):
    """Obtains the hyperparameter settings from an experimental result.

    Checks returned value is hashable.
    """
    rv = list()
    for label in setting_labels:
        rv.append(entry[label])
        hash(rv[-1])
    return collections.namedtuple('Key', setting_labels)(rv)

def summarize_results(results):
    rv = dict()
    for item in results:
        key = get_setting_key(item)
        total_time = sum(item['phase_length'])
        if key not in rv:
            rv[key] = dict()
            rv[key]['total_time_count'] = 1
            rv[key]['total_time_avg'] = total_time
            rv[key]['total_time_second'] = 0
            rv[key]['total_time_min'] = total_time
            rv[key]['total_time_max'] = total_time
        else:
            rv[key]['total_time_count'] += 1
            delta = total_time - rv[key]['total_time_avg']
            rv[key]['total_time_avg'] += delta / rv[key]['total_time_count']
            rv[key]['total_time_second'] += delta * (total_time - rv[key]['total_time_avg'])
            rv[key]['total_time_min'] = min(rv[key]['total_time_min'], total_time)
            rv[key]['total_time_max'] = max(rv[key]['total_time_max'], total_time)
    return rv

def get_best(results):
    rv = dict()
    for key, value in results.items():
        optimizer = key.optimizer
        if (optimizer not in rv) or (value['total_time_avg'] < rv[optimizer][1]['total_time_avg']):
            rv[optimizer] = (key, value)
    return rv

def only_best_results(results, best):
    rv = list()
    for item in results:
        key = get_setting_key(item)
        if key == best[key.optimizer]:
            rv.append(item)
    return rv
