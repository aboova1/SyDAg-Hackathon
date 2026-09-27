"""Site-held-out and randomized trial-block split functions."""

import numpy as np
from sklearn.model_selection import GroupKFold


def outer_sites(data):
    sites = sorted(data.location_key.unique())
    if len(sites) < 3:
        raise ValueError('Need at least three sites for grouped evaluation.')
    for site in sites:
        yield site, data[data.location_key != site].copy(), data[data.location_key == site].copy()


def inner_sites(training, outer_site):
    for site in sorted(training.location_key.unique()):
        yield site, training[training.location_key != site].copy(), training[training.location_key == site].copy()


def _trial_groups(data):
    block = data['block'].astype('string').fillna('unknown-block')
    experiment = data['experiment_key'].astype('string').fillna('unknown-experiment')
    return (data.location_key.astype(str) + '|' + experiment + '|block-' + block).to_numpy()


def outer_random_groups(data, n_splits=5, seed=42):
    """Random outer folds hold each site/experiment/block in one fold."""
    groups = _trial_groups(data)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f'Need at least {n_splits} trial blocks for randomized grouped CV.')
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, groups=groups), 1):
        yield fold, data.iloc[train_idx].copy(), data.iloc[test_idx].copy()


def inner_random_groups(training, n_splits=4, seed=42):
    """Random inner folds hold each site/experiment/block in one fold."""
    groups = _trial_groups(training)
    folds = min(n_splits, len(np.unique(groups)))
    if folds < 2:
        raise ValueError('Need at least two trial blocks for inner validation.')
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(training, groups=groups), 1):
        yield fold, training.iloc[train_idx].copy(), training.iloc[valid_idx].copy()


def randomized_trial_block_split(data, seed=42):
    """Make a site-stratified train/validation/test split by whole trial blocks."""
    groups = _trial_groups(data)
    assignment = {}
    rng = np.random.default_rng(seed)
    for site in sorted(data.location_key.astype(str).unique()):
        site_mask = data.location_key.astype(str).to_numpy() == site
        site_groups = np.unique(groups[site_mask])
        shuffled = rng.permutation(site_groups)
        if len(shuffled) < 2:
            raise ValueError(f'{site} needs at least two trial blocks for a test split.')
        assignment[shuffled[0]] = 'test'
        if len(shuffled) == 2:
            assignment[shuffled[1]] = 'train'
        else:
            assignment[shuffled[1]] = 'validation'
            for group in shuffled[2:]:
                assignment[group] = 'train'

    partitions = np.array([assignment[group] for group in groups], dtype=object)
    split_groups = {
        name: set(groups[partitions == name])
        for name in ('train', 'validation', 'test')
    }
    if any(split_groups[a] & split_groups[b]
           for a, b in (('train', 'validation'), ('train', 'test'),
                        ('validation', 'test'))):
        raise ValueError('A trial block occurs in more than one split.')
    if set.union(*split_groups.values()) != set(groups):
        raise ValueError('The split does not cover every trial block.')
    if any(not np.any(partitions == name)
           for name in ('train', 'validation', 'test')):
        raise ValueError('A train, validation, or test split is empty.')
    return partitions, groups
