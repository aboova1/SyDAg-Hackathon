"""Evaluate within-site yield-rank models for limited scouting capacity."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from .audit import audit
from .dataset import build_dataset
from .model import _candidates
from .random_eval import _fit_predict
from .random_eval import _add_index_trends
from .splits import _trial_groups, inner_sites, outer_sites


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _site_percentile(frame):
    return frame.groupby('location_key').yieldPerAcre.rank(method='average', pct=True)


def _add_site_relative_indices(data, features_path, cutoff, index_columns,
                              plot_table_path=None, include_hybrid=False):
    """Scale image indices within each site using all plots with image data."""
    source = pd.read_csv(features_path, dtype={'plot_key': str},
                         parse_dates=['collection_date'])
    source = source[source.collection_date <= cutoff].copy()
    source = source.sort_values(['plot_key', 'collection_date'])
    latest = source.groupby('plot_key', sort=False).tail(1).copy()
    first = source.groupby('plot_key', sort=False).head(1).set_index('plot_key')
    for name in ('ndvi', 'ndre', 'gndvi'):
        column = f'img_{name}_median'
        latest[f'img_{name}_change'] = latest[column] - latest.plot_key.map(first[column])

    relative_columns, robust_columns, hybrid_columns = [], [], []
    ranked = latest[['plot_key', 'location_key']].copy()
    hybrid_group = None
    if include_hybrid:
        if plot_table_path is None:
            raise ValueError('Hybrid-relative features need the plot table.')
        plots = pd.read_csv(plot_table_path, dtype={'plot_key': str})
        plots = plots[['plot_key', 'season', 'genotype']]
        plots = plots[plots.season == cutoff.year].drop_duplicates('plot_key')
        latest = latest.merge(plots, on='plot_key', how='left', validate='one_to_one')
        latest['genotype'] = latest.genotype.fillna('unknown').astype(str)
        hybrid_group = ['location_key', 'genotype']
        ranked['genotype'] = latest.genotype.to_numpy()
    for column in index_columns:
        if column not in latest:
            raise ValueError(f'Image feature table lacks {column}.')
        relative_column = f'{column}_within_site_pct'
        ranked[relative_column] = latest.groupby('location_key')[column].rank(
            method='average', pct=True).to_numpy()
        relative_columns.append(relative_column)
        median = latest.groupby('location_key')[column].transform('median')
        q1 = latest.groupby('location_key')[column].transform(lambda values: values.quantile(.25))
        q3 = latest.groupby('location_key')[column].transform(lambda values: values.quantile(.75))
        scale = (q3 - q1).replace(0, np.nan)
        robust_column = f'{column}_within_site_robust_z'
        ranked[robust_column] = ((latest[column] - median) / scale).to_numpy()
        robust_columns.append(robust_column)
        if include_hybrid:
            hybrid_column = f'{column}_within_site_hybrid_pct'
            hybrid_rank = latest.groupby(hybrid_group)[column].rank(
                method='average', pct=True)
            hybrid_count = latest.groupby(hybrid_group)[column].transform('count')
            site_rank = latest.groupby('location_key')[column].rank(
                method='average', pct=True)
            ranked[hybrid_column] = hybrid_rank.where(
                hybrid_count >= 3, site_rank).to_numpy()
            hybrid_columns.append(hybrid_column)
    data = data.copy()
    data['plot_key'] = data.plot_key.astype(str)
    ranked['plot_key'] = ranked.plot_key.astype(str)
    result = data.merge(
        ranked[['plot_key'] + relative_columns + robust_columns + hybrid_columns],
        on='plot_key',
                        how='left', validate='one_to_one')
    if len(result) != len(data):
        raise ValueError('Site-relative image join changed the plot count.')
    return result, relative_columns, robust_columns, hybrid_columns


def _rank_metrics(actual, predicted, keys):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keys = np.asarray(keys, dtype=str)
    actual_percentile = pd.Series(actual).rank(method='average', pct=True).to_numpy()
    ranked_prediction = pd.Series(predicted).rank(method='average', pct=True).to_numpy()
    rank_mae = float(mean_absolute_error(actual_percentile, ranked_prediction))
    if np.unique(actual_percentile).size < 2 or np.unique(predicted).size < 2:
        rho = np.nan
    else:
        rho = float(pd.Series(actual_percentile).corr(
            pd.Series(predicted), method='spearman'))
    low = actual <= np.quantile(actual, .2)
    order = np.lexsort((keys, predicted))
    result = []
    for capacity in (.1, .2, .3):
        count = max(1, int(np.ceil(len(actual) * capacity)))
        selected = order[:count]
        low_count = int(low.sum())
        result.append({
            'capacity': capacity,
            'n': len(actual),
            'scout_count': count,
            'low_yield_count': low_count,
            'rank_mae': rank_mae,
            'spearman_rho': rho,
            'low_yield_recall': float(low[selected].sum() / low_count) if low_count else np.nan,
            'low_yield_precision': float(low[selected].mean()),
            'random_recall': count / len(actual),
        })
    return result


def evaluate_rank(root, features, cutoff, output, seed=42, outer_folds=5, inner_folds=4,
                  input_set='fields_indices_plus_site', include_catboost_mae=False,
                  include_catboost_native=False,
                  include_image_trends=False, split_strategy='trial_block',
                  ensemble_selection=False):
    """Fit and assess a site-relative yield-percentile model with grouped CV."""
    root, features, output = Path(root), Path(features), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder for every model run.')
    output.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.year != 2022:
        raise ValueError('The current rank evaluation supports the 2022 development set.')
    if input_set not in {'nitrogen_plus_site', 'fields_plus_site', 'indices_plus_site',
                         'fields_indices_plus_site', 'fields_images_plus_site',
                         'fields_indices_no_genotype',
                         'fields_relative_indices_plus_site',
                         'fields_raw_relative_indices_plus_site',
                         'fields_robust_indices_plus_site',
                         'fields_hybrid_relative_indices_plus_site', 'auto'}:
        raise ValueError('Unknown rank input set.')
    if split_strategy not in {'trial_block', 'held_site'}:
        raise ValueError('split_strategy must be trial_block or held_site.')
    if not audit(root, output / 'source_audit.json')['structural_pass']:
        raise ValueError('The source audit failed. Review source_audit.json before evaluation.')
    feature_audit = json.loads(features.with_suffix('.audit.json').read_text())
    if feature_audit['failed'] or feature_audit['settings']['season'] != 2022:
        raise ValueError('Feature audit failed or feature season does not match.')

    started = time.time()
    data, exclusions, image_cols = build_dataset(root, features, 2022, cutoff)
    trend_columns = []
    if include_image_trends:
        data, trend_columns = _add_index_trends(data, features, cutoff)
    group_ids = _trial_groups(data)
    group_count = len(np.unique(group_ids))
    if split_strategy == 'trial_block' and group_count < outer_folds:
        raise ValueError(f'Need {outer_folds} trial blocks; found {group_count}.')
    if split_strategy == 'held_site' and data.location_key.nunique() != outer_folds:
        raise ValueError('Held-site evaluation needs one outer fold for each site.')
    cores = os.cpu_count() or 1
    field_numeric = ['poundsOfNitrogenPerAcre', 'irrigationProvided', 'planting_day']
    indices = [column for column in image_cols
               if column.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))
               or column in {'img_valid_fraction', 'img_observation_count',
                             'img_age_days', 'img_days_since_first'}]
    index_columns = [column for column in indices
                     if column.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))]
    data, relative_index_columns, robust_index_columns, hybrid_index_columns = (
        _add_site_relative_indices(
            data, features, cutoff, index_columns,
            plot_table_path=root / 'data/processed/plot_records.csv',
            include_hybrid=input_set == 'fields_hybrid_relative_indices_plus_site'))
    image_context_columns = [column for column in indices if column not in index_columns]
    input_sets = {
        'nitrogen_plus_site': (['poundsOfNitrogenPerAcre'], ['location_key']),
        'fields_plus_site': (field_numeric, ['genotype', 'location_key']),
        'indices_plus_site': (indices + trend_columns, ['location_key']),
        'fields_indices_plus_site': (field_numeric + indices + trend_columns,
                                    ['genotype', 'location_key']),
        'fields_indices_no_genotype': (field_numeric + indices + trend_columns,
                                       ['location_key']),
        'fields_images_plus_site': (field_numeric + image_cols,
                                    ['genotype', 'location_key']),
        'fields_relative_indices_plus_site': (
            field_numeric + relative_index_columns + image_context_columns,
            ['genotype', 'location_key']),
        'fields_raw_relative_indices_plus_site': (
            field_numeric + indices + relative_index_columns + trend_columns,
            ['genotype', 'location_key']),
        'fields_robust_indices_plus_site': (
            field_numeric + robust_index_columns + image_context_columns,
            ['genotype', 'location_key']),
        'fields_hybrid_relative_indices_plus_site': (
            field_numeric + hybrid_index_columns + image_context_columns,
            ['genotype', 'location_key']),
    }
    if input_set == 'auto':
        search_input_sets = ('fields_plus_site', 'indices_plus_site',
                             'fields_indices_plus_site',
                             'fields_relative_indices_plus_site')
        auto_models = {'ridge_a1', 'catboost_depth6', 'extra_trees_leaf15'}
        candidate_bank = []
        for candidate_input in search_input_sets:
            candidate_numeric, candidate_categorical = input_sets[candidate_input]
            for candidate_name, _ in _candidates(
                    candidate_numeric, candidate_categorical, cores, seed):
                if candidate_name in auto_models:
                    candidate_bank.append((
                        f'{candidate_input}::{candidate_name}', candidate_input,
                        candidate_name, candidate_numeric, candidate_categorical))
    else:
        numeric, categorical = input_sets[input_set]
        candidate_bank = [
            (name, input_set, name, numeric, categorical)
            for name, _ in _candidates(numeric, categorical, cores, seed,
                                       include_catboost_mae=include_catboost_mae,
                                       include_catboost_native=include_catboost_native)
        ]
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    data[['plot_key', 'location_key', 'experiment_key', 'block', 'collection_date']].to_csv(
        output / 'eligible_plots.csv', index=False)

    y = data.yieldPerAcre.to_numpy()
    outer_groups = _trial_groups(data)
    held_predictions, selection_rows = [], []
    if split_strategy == 'trial_block':
        outer = GroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        outer_splits = [(None, data.iloc[train_idx], data.iloc[test_idx])
                        for train_idx, test_idx in outer.split(data, y, outer_groups)]
    else:
        outer_splits = list(outer_sites(data))
    for outer_fold, (held_site, train, test) in enumerate(outer_splits, 1):
        train_groups = _trial_groups(train)
        if split_strategy == 'trial_block':
            inner_count = min(inner_folds, len(np.unique(train_groups)))
            inner = GroupKFold(n_splits=inner_count, shuffle=True,
                               random_state=seed + outer_fold)
            inner_folds_data = [
                (train.iloc[fit_idx], train.iloc[valid_idx],
                 len(np.unique(train_groups[valid_idx])))
                for fit_idx, valid_idx in inner.split(
                    train, train.yieldPerAcre.to_numpy(), train_groups)
            ]
        else:
            inner_folds_data = [
                (inner_train, valid, len(np.unique(_trial_groups(valid))))
                for _site, inner_train, valid in inner_sites(train, held_site)
            ]
            inner_count = len(inner_folds_data)
        fold_label = f'held site {held_site}' if held_site is not None else 'trial blocks'
        print(f'Outer rank fold {outer_fold}/{len(outer_splits)} ({fold_label}): '
              f'{len(train)} train, {len(test)} test plots, {inner_count} inner folds', flush=True)
        candidate_scores = {}
        for candidate_key, candidate_input, candidate_name, numeric, categorical in candidate_bank:
            recalls, rank_errors = [], []
            for inner_fold, (inner_train, valid, validation_blocks) in enumerate(
                    inner_folds_data, 1):
                target_rank = _site_percentile(inner_train).to_numpy()
                selection_seeds = (42, 101, 2026) if ensemble_selection else (seed,)
                seed_predictions = []
                for model_seed in selection_seeds:
                    estimator = dict(_candidates(
                        numeric, categorical, cores, model_seed,
                        include_catboost_mae=(include_catboost_mae
                                              if input_set != 'auto' else False),
                        include_catboost_native=(include_catboost_native
                                                 if input_set != 'auto' else False)))[candidate_name]
                    seed_predictions.append(_fit_predict(
                        estimator, candidate_name, inner_train, target_rank, valid))
                predicted_rank = np.mean(seed_predictions, axis=0)
                site_rows = []
                for site, group in valid.assign(_predicted_rank=predicted_rank).groupby('location_key'):
                    score = _rank_metrics(group.yieldPerAcre, group['_predicted_rank'], group.plot_key)
                    at_20 = next(row for row in score if row['capacity'] == .2)
                    site_rows.append(at_20)
                recall = float(np.mean([row['low_yield_recall'] for row in site_rows]))
                rank_error = float(np.mean([row['rank_mae'] for row in site_rows]))
                recalls.append(recall)
                rank_errors.append(rank_error)
                selection_rows.append({
                    'outer_fold': outer_fold, 'inner_fold': inner_fold,
                    'held_out_site': held_site,
                    'input_set': candidate_input,
                    'estimator': candidate_name, 'equal_site_recall20': recall,
                    'equal_site_rank_mae': rank_error, 'validation_plots': len(valid),
                    'validation_blocks': validation_blocks,
                })
            candidate_scores[candidate_key] = {
                'recall': float(np.mean(recalls)), 'rank_mae': float(np.mean(rank_errors))}

        chosen = min(candidate_scores, key=lambda name: (
            -candidate_scores[name]['recall'], candidate_scores[name]['rank_mae'], name))
        chosen_key, chosen_input, chosen_model, numeric, categorical = next(
            row for row in candidate_bank if row[0] == chosen)
        train_rank = _site_percentile(train).to_numpy()
        prediction_seeds = {}
        for model_seed in (42, 101, 2026):
            estimator = dict(_candidates(
                numeric, categorical, cores, model_seed,
                include_catboost_mae=(include_catboost_mae
                                      if input_set != 'auto' else False),
                include_catboost_native=(include_catboost_native
                                         if input_set != 'auto' else False)))[chosen_model]
            prediction_seeds[model_seed] = _fit_predict(
                estimator, chosen_model, train, train_rank, test)
        held_predictions.append(pd.DataFrame({
            'plot_key': test.plot_key.astype(str).to_numpy(),
            'location_key': test.location_key.to_numpy(),
            'held_out_site': held_site,
            'experiment_key': test.experiment_key.to_numpy(),
            'block': test.block.to_numpy(), 'fold': outer_fold,
            'actual_yield': test.yieldPerAcre.to_numpy(),
            'predicted_percentile_seed_42': prediction_seeds[42],
            'predicted_percentile_seed_101': prediction_seeds[101],
            'predicted_percentile_seed_2026': prediction_seeds[2026],
            'predicted_percentile_seed_mean': np.mean(
                [prediction_seeds[42], prediction_seeds[101], prediction_seeds[2026]], axis=0),
            'predicted_percentile_seed_median': np.median(
                [prediction_seeds[42], prediction_seeds[101], prediction_seeds[2026]], axis=0),
            'selected_model': chosen_model, 'selected_input_set': chosen_input,
        }))
        pd.concat(held_predictions, ignore_index=True).to_csv(
            output / 'predictions.partial.csv', index=False)
        pd.DataFrame(selection_rows).to_csv(output / 'inner_selection_scores.partial.csv', index=False)
        print(f'  selected {chosen_input} / {chosen_model}; inner recall20 '
              f'{candidate_scores[chosen]["recall"]:.3f}; rank MAE '
              f'{candidate_scores[chosen]["rank_mae"]:.3f}', flush=True)

    predictions = pd.concat(held_predictions, ignore_index=True)
    if predictions.duplicated('plot_key').any():
        raise ValueError('Each plot must have one out-of-fold rank prediction.')
    metrics = []
    prediction_columns = (
        'predicted_percentile_seed_42', 'predicted_percentile_seed_101',
        'predicted_percentile_seed_2026', 'predicted_percentile_seed_mean',
        'predicted_percentile_seed_median')
    for site, group in predictions.groupby('location_key'):
        for column in prediction_columns:
            for score in _rank_metrics(group.actual_yield, group[column], group.plot_key):
                metrics.append({'site': site, 'fold': 'all_oof',
                                'model_seed': column.rsplit('_', 1)[1], **score})
    detail = data[['plot_key', 'genotype', 'row', 'range', 'collection_date',
                   'img_age_days', 'img_observation_count']]
    predictions = predictions.merge(detail, on='plot_key', validate='one_to_one')
    metrics = pd.DataFrame(metrics)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(selection_rows).to_csv(output / 'inner_selection_scores.csv', index=False)
    aggregate = metrics.groupby(['model_seed', 'capacity']).agg(
        equal_site_rank_mae=('rank_mae', 'mean'),
        equal_site_spearman=('spearman_rho', 'mean'),
        equal_site_low_yield_recall=('low_yield_recall', 'mean'),
        equal_site_low_yield_precision=('low_yield_precision', 'mean'),
        equal_site_random_recall=('random_recall', 'mean'),
    ).reset_index()
    aggregate.to_csv(output / 'aggregate_metrics.csv', index=False)
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = 'unknown'
    metadata = {
        'schema_version': 1, 'season': 2022, 'cutoff': cutoff.date().isoformat(),
        'target': 'within-site yield percentile; lower rank means higher scouting priority',
        'split_strategy': ('nested randomized GroupKFold by location, experiment, and block'
                           if split_strategy == 'trial_block'
                           else 'nested leave-one-site-out validation'),
        'split_strategy_key': split_strategy,
        'selection_objective': 'equal-site low-yield recall at 20% capacity',
        'input_set': input_set,
        'image_index_transform': ('nested selection across raw and site-percentile inputs'
                                  if input_set == 'auto'
                                  else 'within-site and hybrid percentile ranks; groups with fewer than three plots use site ranks'
                                  if input_set == 'fields_hybrid_relative_indices_plus_site'
                                  else 'within-site robust median/IQR scales from all image-eligible plots, without yield labels'
                                  if input_set == 'fields_robust_indices_plus_site'
                                  else 'raw and within-site percentile ranks from all image-eligible plots, without yield labels'
                                  if input_set == 'fields_raw_relative_indices_plus_site'
                                  else 'within-site percentile ranks from all image-eligible plots, without yield labels'
                                  if input_set == 'fields_relative_indices_plus_site'
                                  else 'raw extracted summaries'),
        'include_catboost_mae': include_catboost_mae,
        'include_catboost_native': include_catboost_native,
        'include_image_trends': include_image_trends,
        'ensemble_selection': ensemble_selection,
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seed': seed,
        'group_count': group_count, 'eligible_plots': len(data),
        'sites': sorted(data.location_key.unique()), 'cores_used': cores,
        'base_models': [{'input_set': input_name, 'model': model_name}
                        for _key, input_name, model_name, _numeric, _categorical
                        in candidate_bank],
        'limitations': ('This ranks plots within known sites in one season. It does not predict '
                        'yield in bu/acre or test new seasons.' if split_strategy == 'trial_block'
                        else 'This tests transfer to one held-out site within the 2022 season. '
                             'It does not test a future season or provide yield in bu/acre.'),
        'feature_hash': _hash(features),
        'plot_table_hash': _hash(root / 'data/processed/plot_records.csv'),
        'code_revision': revision, 'python': platform.python_version(),
        'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
        'elapsed_seconds': round(time.time() - started, 1),
    }
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'predictions.partial.csv').unlink(missing_ok=True)
    (output / 'inner_selection_scores.partial.csv').unlink(missing_ok=True)
    print(f'Saved rank evaluation to {output}', flush=True)
