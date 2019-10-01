# -*- coding: utf-8 -*-

import collections
import json
import numpy as np
import pandas as pd

setting_labels = [
    'architecture',
    'beta_1',
    'beta_2',
    'criteria',
    'dataset',
    'digits',
    'fold_count',
    'has_all_digits_phase',
    'log_frequency',
    'lr',
    'minimum_steps',
    'momentum',
    'optimizer',
    'phases',
    'required_accuracy',
    'rho',
    'steps',
    'tolerance']
repeat_labels = [
    'seed',
    'test_folds',
    'train_folds',
    'validation_folds']
result_labels = [
    'accuracies',
    'correct',
    'end_time',
    'outfile',
    'phase_length',
    'predictions',
    'start_time',
    'success']

Key = collections.namedtuple('Key', setting_labels)

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
    return Key(* rv)

def phase_length_summary(results):
    """Obtains summary statistics for phase lengths over multiple seeds and folds."""
    rv = dict()
    for _, row in results.iterrows():
        key = get_setting_key(row)
        total_time = sum(row['phase_length'])
        if key not in rv:
            rv[key] = dict()
            rv[key]['count'] = 1
            rv[key]['total_time_avg'] = total_time
            rv[key]['total_time_second'] = 0
            rv[key]['total_time_min'] = total_time
            rv[key]['total_time_max'] = total_time
            rv[key]['phases_avg'] = np.array(row['phase_length'])
            rv[key]['phases_second'] = np.zeros(len(row['phase_length']))
            rv[key]['phases_min'] = np.array(row['phase_length'])
            rv[key]['phases_max'] = np.array(row['phase_length'])
        else:
            rv[key]['count'] += 1
            delta = total_time - rv[key]['total_time_avg']
            rv[key]['total_time_avg'] += delta / rv[key]['count']
            rv[key]['total_time_second'] += delta * (total_time - rv[key]['total_time_avg'])
            rv[key]['total_time_min'] = min(rv[key]['total_time_min'], total_time)
            rv[key]['total_time_max'] = max(rv[key]['total_time_max'], total_time)
            for i, phase_length in enumerate(row['phase_length']):
                delta = phase_length - rv[key]['phases_avg'][i]
                rv[key]['phases_avg'][i] += delta / rv[key]['count']
                rv[key]['phases_second'][i] += delta * (phase_length - rv[key]['phases_avg'][i])
                rv[key]['phases_min'][i] = min(rv[key]['phases_min'][i], phase_length)
                rv[key]['phases_max'][i] = max(rv[key]['phases_max'][i], phase_length)
    for key in rv.keys():
        rv[key]['total_time_var'] = rv[key]['total_time_second'] / rv[key]['count']
        rv[key]['total_time_sem'] = np.sqrt(rv[key]['total_time_var'] / rv[key]['count'])
        rv[key]['phases_var'] = rv[key]['phases_second'] / rv[key]['count']
        rv[key]['phases_sem'] = np.sqrt(rv[key]['phases_var'] / rv[key]['count'])
        del rv[key]['total_time_second']
        del rv[key]['phases_second']
    return rv

def cumulative_errors_summary(results):
    """Obtains summary statistics for cumulative errors over multiple seeds and folds. Assumes that
    phases have a fixed length.
    """
    rv = dict()
    for _, row in results.iterrows():
        key = get_setting_key(row)
        c_errors = np.cumsum(1 - np.array(row['correct'], dtype=int))
        total_errors = c_errors[-1]
        if key not in rv:
            rv[key] = dict()
            rv[key]['count'] = np.ones(len(c_errors))
            rv[key]['total_errors_avg'] = total_errors
            rv[key]['total_errors_second'] = 0
            rv[key]['total_errors_min'] = total_errors
            rv[key]['total_errors_max'] = total_errors
            rv[key]['c_errors_avg'] = np.array(c_errors)
            rv[key]['c_errors_second'] = np.zeros(len(c_errors))
            rv[key]['c_errors_min'] = np.array(c_errors)
            rv[key]['c_errors_max'] = np.array(c_errors)
        else:
            if len(c_errors) > len(rv[key]['count']):
                new_count = np.ones(len(c_errors))
                new_count[:len(rv[key]['count'])] += rv[key]['count']
                rv[key]['count'] = new_count
            else:
                rv[key]['count'][:len(c_errors)] += 1
            delta = total_errors - rv[key]['total_errors_avg']
            rv[key]['total_errors_avg'] += delta / rv[key]['count']
            rv[key]['total_errors_second'] += delta * (total_errors - rv[key]['total_errors_avg'])
            rv[key]['total_errors_min'] = min(rv[key]['total_errors_min'], total_errors)
            rv[key]['total_errors_max'] = max(rv[key]['total_errors_max'], total_errors)
            for i, c in enumerate(c_errors):
                delta = c - rv[key]['c_errors_avg'][i]
                rv[key]['c_errors_avg'][i] += delta / rv[key]['count'][i]
                rv[key]['c_errors_second'][i] += delta * (c - rv[key]['c_errors_avg'][i])
                rv[key]['c_errors_min'][i] = min(rv[key]['c_errors_min'][i], c)
                rv[key]['c_errors_max'][i] = max(rv[key]['c_errors_max'][i], c)
    for key in rv.keys():
        rv[key]['total_errors_var'] = rv[key]['total_errors_second'] / rv[key]['count'][0]
        rv[key]['total_errors_sem'] = np.sqrt(rv[key]['total_errors_var'] / rv[key]['count'][0])
        rv[key]['c_errors_var'] = rv[key]['c_errors_second'] / rv[key]['count']
        rv[key]['c_errors_sem'] = np.sqrt(rv[key]['c_errors_var'] / rv[key]['count'])
        del rv[key]['total_errors_second']
        del rv[key]['c_errors_second']
    return rv

def get_best(summary, metric):
    """Filters a summary to obtain the best average total of a metric for each optimizer."""
    if metric == 'time':
        metric_key = 'total_time_avg'
    elif metric == 'errors':
        metric_key = 'total_errors_avg'
    else:
        assert(False)
    rv = dict()
    for key, value in summary.items():
        optimizer = key.optimizer
        if (optimizer not in rv) or (value[metric_key] < rv[optimizer][1][metric_key]):
            rv[optimizer] = (key, value)
    return rv

def only_best(results, best):
    """Filters a set of results to only contain the entries that used the hyperparameter setting
    with the best average total for some metric with the respective optimizer.
    """
    rv = pd.DataFrame(columns=results.columns)
    for _, row in results.iterrows():
        key = get_setting_key(row)
        if key == best[key.optimizer][0]:
            rv = rv.append(row)
    return rv
