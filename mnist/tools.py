# -*- coding: utf-8 -*-

import collections
import json
import numpy as np
import pandas as pd

infrastructure_labels = [
    'end_time',
    'outfile',
    'start_time']
repeat_labels = [
    'init_seed',
    'shuffle_seed',
    'test_folds',
    'train_folds',
    'validation_folds']
result_labels = [
    'accuracies',
    'activation_overlap',
    'correct',
    'phase_length',
    'predictions',
    'sparse_activation_overlap',
    'success']
setting_labels = [
    'beta_1',
    'beta_2',
    'criteria',
    'dataset',
    'digits',
    'fold_count',
    'hold_steps',
    'log_frequency',
    'lr',
    'minimum_steps',
    'momentum',
    'optimizer',
    'phases',
    'required_accuracy',
    'rho',
    'steps',
    'test_on_all_digits',
    'tolerance']

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
        if isinstance(entry[label], float) and np.isnan(entry[label]):
            rv.append(None)
        else:
            rv.append(entry[label])
        hash(rv[-1])
    return Key(* rv)

def get_summary(results):
    """Obtains select summary statistics over folds and seeds for some results set."""
    rv = dict()
    for _, row in results.iterrows():
        key = get_setting_key(row)
        errors = np.cumsum(1 - np.array(row['correct'], dtype=int))
        total_errors = errors[-1]
        total_time = sum(row['phase_length'])
        if key not in rv:
            rv[key] = dict()
            rv[key]['count'] = np.ones(len(errors))
            rv[key]['total_errors_avg'] = total_errors
            rv[key]['total_errors_second'] = 0
            rv[key]['total_errors_min'] = total_errors
            rv[key]['total_errors_max'] = total_errors
            rv[key]['total_time_avg'] = total_time
            rv[key]['total_time_second'] = 0
            rv[key]['total_time_min'] = total_time
            rv[key]['total_time_max'] = total_time
            rv[key]['phases_avg'] = np.array(row['phase_length'])
            rv[key]['phases_second'] = np.zeros(len(row['phase_length']))
            rv[key]['phases_min'] = np.array(row['phase_length'])
            rv[key]['phases_max'] = np.array(row['phase_length'])
            rv[key]['errors_avg'] = np.array(errors)
            rv[key]['errors_second'] = np.zeros(len(errors))
            rv[key]['errors_min'] = np.array(errors)
            rv[key]['errors_max'] = np.array(errors)
            rv[key].update(dict(Key._asdict(key)))
        else:
            if len(errors) > len(rv[key]['count']):
                new_count = np.zeros(len(errors))
                new_avg = np.zeros(len(errors))
                new_second = np.zeros(len(errors))
                new_min = np.ones(len(errors)) * np.nan
                new_max = np.ones(len(errors)) * np.nan
                np.copyto(new_count[:len(rv[key]['count'])], rv[key]['count'])
                np.copyto(new_avg[:len(rv[key]['errors_avg'])], rv[key]['errors_avg'])
                np.copyto(new_second[:len(rv[key]['errors_second'])], rv[key]['errors_second'])
                np.copyto(new_min[:len(rv[key]['errors_min'])], rv[key]['errors_min'])
                np.copyto(new_max[:len(rv[key]['errors_max'])], rv[key]['errors_max'])
                rv[key]['count'] = new_count
                rv[key]['errors_avg'] = new_avg
                rv[key]['errors_second'] = new_second
                rv[key]['errors_min'] = new_min
                rv[key]['errors_max'] = new_max
            rv[key]['count'][:len(errors)] += 1
            delta = total_errors - rv[key]['total_errors_avg']
            rv[key]['total_errors_avg'] += delta / rv[key]['count'][0]
            rv[key]['total_errors_second'] += delta * (total_errors - rv[key]['total_errors_avg'])
            rv[key]['total_errors_min'] = min(rv[key]['total_errors_min'], total_errors)
            rv[key]['total_errors_max'] = max(rv[key]['total_errors_max'], total_errors)
            delta = total_time - rv[key]['total_time_avg']
            rv[key]['total_time_avg'] += delta / rv[key]['count'][0]
            rv[key]['total_time_second'] += delta * (total_time - rv[key]['total_time_avg'])
            rv[key]['total_time_min'] = min(rv[key]['total_time_min'], total_time)
            rv[key]['total_time_max'] = max(rv[key]['total_time_max'], total_time)
            for i, v in enumerate(errors):
                delta = v - rv[key]['errors_avg'][i]
                rv[key]['errors_avg'][i] += delta / rv[key]['count'][i]
                rv[key]['errors_second'][i] += delta * (v - rv[key]['errors_avg'][i])
                rv[key]['errors_min'][i] = min(rv[key]['errors_min'][i], v)
                rv[key]['errors_max'][i] = max(rv[key]['errors_max'][i], v)
            for i, v in enumerate(row['phase_length']):
                delta = v - rv[key]['phases_avg'][i]
                rv[key]['phases_avg'][i] += delta / rv[key]['count'][0]
                rv[key]['phases_second'][i] += delta * (v - rv[key]['phases_avg'][i])
                rv[key]['phases_min'][i] = min(rv[key]['phases_min'][i], v)
                rv[key]['phases_max'][i] = max(rv[key]['phases_max'][i], v)
    for key in rv.keys():
        rv[key]['total_errors_var'] = rv[key]['total_errors_second'] / rv[key]['count'][0]
        rv[key]['total_errors_sem'] = np.sqrt(rv[key]['total_errors_var'] / rv[key]['count'][0])
        rv[key]['total_time_var'] = rv[key]['total_time_second'] / rv[key]['count'][0]
        rv[key]['total_time_sem'] = np.sqrt(rv[key]['total_time_var'] / rv[key]['count'][0])
        rv[key]['errors_var'] = rv[key]['errors_second'] / rv[key]['count']
        rv[key]['errors_sem'] = np.sqrt(rv[key]['errors_var'] / rv[key]['count'])
        rv[key]['phases_var'] = rv[key]['phases_second'] / rv[key]['count'][0]
        rv[key]['phases_sem'] = np.sqrt(rv[key]['phases_var'] / rv[key]['count'][0])
        del rv[key]['total_errors_second']
        del rv[key]['total_time_second']
        del rv[key]['errors_second']
        del rv[key]['phases_second']
    return rv

def total_time_metric(buffer_count=0, buffer_value=0):
    return lambda x: x['total_time_avg'] + buffer_value * max(buffer_count - x['count'][0], 0)

def errors_metric():
    return lambda x: x['total_errors_avg']

def phase_time_metric(phase, buffer_count=0, buffer_value=0):
    return lambda x: sum(x['phases_avg'][:phase] +
                         buffer_value * max(buffer_count - x['count'][0], 0))

def get_best(summary, metric):
    """Filters a summary to obtain the best average total of a metric for each optimizer."""
    rv = dict()
    for key, value in summary.items():
        optimizer = key.optimizer
        if (optimizer not in rv) or (metric(value) < metric(rv[optimizer])):
            rv[optimizer] = value
    return rv

def get_only_best(results, best):
    """Filters a set of results to only contain the entries that used the hyperparameter setting
    with the best average total for some metric with the respective optimizer.
    """
    rv = pd.DataFrame(columns=results.columns)
    for _, row in results.iterrows():
        key = get_setting_key(row)
        if key == get_setting_key(best[key.optimizer]):
            rv = rv.append(row)
    return rv
