# -*- coding: utf-8 -*-

import collections
import json
import numpy as np
import pandas as pd

# prevent recursive imports
GOAL_POSITION = 0.5
OBSERVATION_MAX = (0.6, 0.07)
OBSERVATION_MIN = (- 1.2, - 0.07)

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
    'pairwise_interference',
    'sparse_activation_overlap']
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
    rv = list()
    for i in range(len(value)):
        rv.append(scale(value[i], OBSERVATION_MIN[i], OBSERVATION_MAX[i], - 1, 1))
    assert(all([- 1 <= rv[i] <= 1 for i in range(len(rv))]))
    return tuple(rv)

def load_clean_data(files):
    rv = list()
    for filename in files:
        with open(filename, 'r') as infile:
            rv += json.load(infile)

    # clean up data to fit assumptions
    EPS = 1e-9
    for entry in rv:
        entry['auc'] = sum(entry['accuracy'])
        entry['final_accuracy'] = entry['accuracy'][-1]
        entry['lambda_'] = entry['lambda']  # pandas doesn't allow for columns to be called lambda
        del entry['lambda']
        if entry['approximator'] == 'constant':
            entry['optimizer'] = 'constant'  # rest of plotting relies on optimizer being defined
        else:
            entry['final_activation_overlap'] = entry['activation_overlap'][-1]
            entry['final_sparse_activation_overlap'] = entry['sparse_activation_overlap'][-1]
            entry['final_pairwise_interference'] = entry['pairwise_interference'][-1]
        if (entry['optimizer'] == 'sgd') and (abs(float(entry['momentum'])) > EPS):
            entry['optimizer'] = 'momentum'  # separate sgd with and without momentum

    to_delete = list()
    constant_seeds = set()
    for i in range(len(rv)):
        if rv[i]['approximator'] == 'constant':
            if rv[i]['env_seed'] in constant_seeds:
                to_delete.append(i)
            else:
                constant_seeds.add(rv[i]['env_seed'])
    for i in reversed(to_delete):
        del rv[i]

    # return cleaned data
    return rv

def get_summary(data):
    """Obtains select summary statistics over seeds for some results set."""
    temp = dict()
    for entry in data:
        key = get_hyperparameter_key(entry)
        if key in temp:
            temp[key]['auc'].append(entry['auc'])
            temp[key]['mean_accuracy'].append(np.mean(entry['accuracy']))
            temp[key]['final_accuracy'].append(entry['final_accuracy'])
            if key.optimizer != 'constant':
                for key2 in ['activation_overlap', 'sparse_activation_overlap', 'pairwise_interference']:
                    temp[key]['final_{}'.format(key2)].append(entry['final_{}'.format(key2)])
                    temp[key]['mean_{}'.format(key2)].append(np.mean(entry[key2]))
        else:
            value = dict()
            value['auc'] = [entry['auc']]
            value['mean_accuracy'] = [np.mean(entry['accuracy'])]
            value['final_accuracy'] = [entry['final_accuracy']]
            if key.optimizer != 'constant':
                for metric in ['activation_overlap', 'sparse_activation_overlap', 'pairwise_interference']:
                    value['final_{}'.format(metric)] = [entry['final_{}'.format(metric)]]
                    value['mean_{}'.format(metric)] = [np.mean(entry[metric])]
            temp[key] = value
    table = list()
    for key1, value in temp.items():
        entry = dict(key1._asdict())
        entry['count'] = len(value['auc'])
        for key2 in ['auc',
                     'mean_accuracy',
                     'final_accuracy',
                     'mean_activation_overlap',
                     'mean_sparse_activation_overlap',
                     'mean_pairwise_interference']:
            try:
                entry['{}_mean'.format(key2)] = np.mean(value[key2])
                entry['{}_stderr'.format(key2)] = np.std(value[key2]) / np.sqrt(len(value[key2]))
            except KeyError:
                entry['{}_mean'.format(key2)] = np.nan
                entry['{}_stderr'.format(key2)] = np.nan
        table.append(entry)
    return pd.DataFrame(list_of_dicts_to_dict_of_lists(table))

def get_best(data, metric, summary=None):
    if summary is None:
        summary = get_summary(data)
    assert(metric in ['auc', 'final_accuracy'])

    # build best table
    best = list()
    for optimizer in summary['optimizer'].unique():
        sub_table = summary[summary['optimizer'] == optimizer]
        best.append((sub_table.loc[sub_table['{}_mean'.format(metric)].idxmin()]).to_dict())
    return pd.DataFrame(list_of_dicts_to_dict_of_lists(best))

def get_best_by_optimizer(data, best_auc_table=None):
    if best_auc_table is None:
        best_auc_table = get_best_auc_table(data, get_auc_table(data))

    # build best by optimizer dict
    best_keys = {row['optimizer']: get_hyperparameter_key(row.to_dict()) for _, row in best_auc_table.iterrows()}
    rv = dict()
    for entry in data:
        optimizer = entry['optimizer']
        if get_hyperparameter_key(entry) == best_keys[optimizer]:
            if optimizer not in rv:
                rv[optimizer] = [entry]
            else:
                rv[optimizer].append(entry)
    return rv

def get_best_by_optimizer_summary(data, best_auc_table=None, best_by_optimizer=None):
    if best_auc_table is None:
        best_auc_table = get_best(data, 'auc')
    if best_by_optimizer is None:
        best_by_optimizer = get_best_by_optimizer(data, best_auc_table)

    # build best by optimizer summary dict
    rv = {row['optimizer']: row.to_dict() for _, row in best_auc_table.iterrows()}
    for key1 in rv.keys():
        for key2 in result_labels:
            try:
                values = np.array([item[key2] for item in best_by_optimizer[key1]])
                rv[key1][key2 + '_mean'] = np.mean(values, axis=0)
                rv[key1][key2 + '_stderr'] = np.std(values, axis=0) / np.sqrt(values.shape[0])
            except TypeError:
                pass
    return rv
