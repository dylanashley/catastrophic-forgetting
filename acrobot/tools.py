# -*- coding: utf-8 -*-

import collections
import itertools
import json
import numpy as np
import pandas as pd

# prevent recursive imports
OBSERVATION_MAX = np.array([1.0, 1.0, 1.0, 1.0, 4 * np.pi, 9 * np.pi])
OBSERVATION_MIN = - OBSERVATION_MAX

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
    'activation_similarity',
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
    rv = list()
    for i in range(len(value)):
        rv.append(scale(value[i], OBSERVATION_MIN[i], OBSERVATION_MAX[i], - 1, 1))
    assert(all([- 1 <= rv[i] <= 1 for i in range(len(rv))]))
    return tuple(rv)

def load_data(files):
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
            entry['final_activation_similarity'] = entry['activation_similarity'][-1]
            entry['final_pairwise_interference'] = entry['pairwise_interference'][-1]
        if entry['optimizer'] == 'sgd':  # separate sgd with and without momentum
            if abs(float(entry['momentum'])) > EPS:
                entry['optimizer'] = 'momentum'
            else:
                entry['momentum'] = None

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
            temp[key]['accuracy'].append(entry['accuracy'])
            if key.optimizer != 'constant':
                for key2 in ['activation_similarity', 'pairwise_interference']:
                    temp[key]['final_{}'.format(key2)].append(entry['final_{}'.format(key2)])
                    temp[key]['mean_{}'.format(key2)].append(np.mean(entry[key2]))
                    temp[key]['{}'.format(key2)].append(entry[key2])
        else:
            value = dict()
            value['auc'] = [entry['auc']]
            value['mean_accuracy'] = [np.mean(entry['accuracy'])]
            value['final_accuracy'] = [entry['final_accuracy']]
            value['accuracy'] = [entry['accuracy']]
            if key.optimizer != 'constant':
                for metric in ['activation_similarity', 'pairwise_interference']:
                    value['final_{}'.format(metric)] = [entry['final_{}'.format(metric)]]
                    value['mean_{}'.format(metric)] = [np.mean(entry[metric])]
                    value['{}'.format(metric)] = [entry[metric]]
            temp[key] = value
    table = list()
    for key1, value in temp.items():
        entry = dict(key1._asdict())
        entry['count'] = len(value['auc'])
        for key2 in ['auc',
                     'mean_accuracy',
                     'mean_activation_similarity',
                     'mean_pairwise_interference',
                     'final_accuracy',
                     'final_activation_similarity',
                     'final_pairwise_interference',
                     'accuracy',
                     'activation_similarity',
                     'pairwise_interference']:
            try:
                entry['{}_mean'.format(key2)] = np.mean(value[key2], axis=0)
                entry['{}_stderr'.format(key2)] = np.std(value[key2], axis=0) / np.sqrt(np.shape(value[key2])[0])
            except KeyError:
                entry['{}_mean'.format(key2)] = np.nan
                entry['{}_stderr'.format(key2)] = np.nan
        table.append(entry)
    return pd.DataFrame(list_of_dicts_to_dict_of_lists(table))

def get_subtable(df, fields, values):
    assert(len(fields) == len(values))
    rv = df
    for field, value in zip(fields, values):
        try:  # have to deal with nan != nan
            isnan = np.isnan(value)
        except TypeError:
            isnan = False

        if (value in rv[field].unique()) or (isnan and np.isnan(rv[field].unique()).any()):
            if isnan:
                rv = rv[np.isnan(rv[field])]
            else:
                rv = rv[rv[field] == value]
        else:
            return pd.DataFrame({field: [] for field in fields})
    return rv

def get_unique(df, fields):
    rv = [list(df[field].unique()) for field in fields]
    rv = list(itertools.product(*rv))
    return [v for v in rv if len(get_subtable(df, fields, v)) > 0]

def get_best(data, fields, metric, summary=None):
    if summary is None:
        summary = get_summary(data)
    assert(metric in ['auc', 'final_accuracy'])

    # build best table
    rv = list()
    for key in get_unique(summary, fields):
        subtable = get_subtable(summary, fields, key)
        rv.append((subtable.loc[subtable['{}_mean'.format(metric)].idxmin()]).to_dict())
    return pd.DataFrame(list_of_dicts_to_dict_of_lists(rv))
