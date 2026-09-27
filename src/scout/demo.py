"""Prepare a demo run that joins direct rank priorities to yield estimates."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_rank_demo(rank_run, yield_run, output):
    rank_run, yield_run, output = map(Path, (rank_run, yield_run, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output folder for the rank demo.')
    rank_meta = json.loads((rank_run / 'run.json').read_text())
    yield_meta = json.loads((yield_run / 'run.json').read_text())
    if rank_meta.get('season') != 2022 or yield_meta.get('season') != 2022:
        raise ValueError('The rank demo requires 2022 runs.')
    if rank_meta.get('cutoff') != yield_meta.get('cutoff'):
        raise ValueError('Rank and yield runs must use the same image cutoff.')
    if rank_meta.get('input_set', 'fields_indices_plus_site') != 'fields_indices_plus_site':
        raise ValueError('The rank run must use field records and image indices.')

    rank = pd.read_csv(rank_run / 'predictions.csv')
    regression = pd.read_csv(yield_run / 'predictions.csv')
    regression = regression[(regression.input_set == 'fields_indices_plus_site')
                            & (regression.capacity == .2)].drop_duplicates('plot_key')
    if rank.plot_key.duplicated().any() or regression.plot_key.duplicated().any():
        raise ValueError('Each source run must have one row per plot.')
    if set(rank.plot_key.astype(str)) != set(regression.plot_key.astype(str)):
        raise ValueError('Rank and yield runs have different plot sets.')
    rank = rank.set_index(rank.plot_key.astype(str)).sort_index()
    regression = regression.set_index(regression.plot_key.astype(str)).loc[rank.index]
    if not np.array_equal(rank.fold.to_numpy(), regression.fold.to_numpy()):
        raise ValueError('Rank and yield runs have different block folds.')
    if not np.allclose(rank.actual_yield, regression.actual_yield):
        raise ValueError('Rank and yield runs have different harvest labels.')

    base = regression.reset_index(drop=True).copy()
    rank_seed_columns = [f'predicted_percentile_seed_{seed}'
                         for seed in (42, 101, 2026)]
    base['priority_score'] = rank[rank_seed_columns].mean(axis=1).to_numpy()
    base['rank_model_seed_42'] = rank[rank_seed_columns[0]].to_numpy()
    base['rank_model_seed_101'] = rank[rank_seed_columns[1]].to_numpy()
    base['rank_model_seed_2026'] = rank[rank_seed_columns[2]].to_numpy()
    base['rank_selected_model'] = rank.selected_model.to_numpy()
    base['yield_estimator'] = base.estimator
    base['estimator'] = base.rank_selected_model
    base['input_set'] = 'direct_rank_priority'
    base['priority_method'] = 'Direct within-site yield percentile'
    output_rows = []
    for site, site_rows in base.groupby('location_key', sort=True):
        keys = site_rows.plot_key.astype(str).to_numpy()
        score = site_rows.priority_score.to_numpy()
        order = np.lexsort((keys, score))
        rank_value = np.empty(len(site_rows), dtype=int)
        rank_value[order] = np.arange(1, len(site_rows) + 1)
        seed_ranks = {}
        demo_seed_columns = ['rank_model_seed_42', 'rank_model_seed_101',
                             'rank_model_seed_2026']
        for seed_column in demo_seed_columns:
            values = site_rows[seed_column].to_numpy()
            seed_ranks[seed_column] = np.lexsort((keys, values))
        for capacity in (.1, .2, .3):
            count = max(1, int(np.ceil(len(site_rows) * capacity)))
            selected_keys = set(keys[order[:count]])
            seed_selected = [set(keys[seed_ranks[column][:count]])
                             for column in demo_seed_columns]
            frequency = {key: sum(key in chosen for chosen in seed_selected) / 3
                         for key in keys}
            rows = site_rows.copy()
            rows['capacity'] = capacity
            rows['rank'] = rank_value
            rows['selected'] = rows.plot_key.astype(str).isin(selected_keys)
            rows['visit_frequency'] = rows.plot_key.astype(str).map(frequency)
            output_rows.append(rows)
    predictions = pd.concat(output_rows, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics = pd.read_csv(rank_run / 'metrics.csv')
    metrics.to_csv(output / 'metrics.csv', index=False)
    aggregate = pd.read_csv(rank_run / 'aggregate_metrics.csv')
    aggregate.to_csv(output / 'aggregate_metrics.csv', index=False)
    metadata = dict(rank_meta)
    metadata.update({
        'demo_ready': True,
        'display_protocol': 'Randomized trial-block holdout at known sites',
        'priority_method': 'Direct within-site yield percentile',
        'yield_estimate_source': str(yield_run),
        'priority_score_note': 'The priority score sets visit order. The bu/acre value comes from a separate yield model.',
        'source_rank_run': str(rank_run),
        'source_yield_run': str(yield_run),
    })
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared direct-rank demo at {output}')
