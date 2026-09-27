"""Compare scouting inputs on separate randomized trial-block data."""

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

from .audit import audit
from .dataset import build_dataset, build_dataset_by_dap
from .model import _candidates
from .random_eval import _fit_predict
from .rank_eval import _rank_metrics, _site_percentile
from .splits import randomized_trial_block_split


MODEL_SEEDS = (42, 101, 2026)
CODE_FILES = (
    'src/scout/cli.py',
    'src/scout/rank_input_split_eval.py',
    'src/scout/dataset.py',
    'src/scout/rank_eval.py',
    'src/scout/model.py',
    'src/scout/random_eval.py',
    'src/scout/splits.py',
    'src/scout/audit.py',
)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _site_scores(frame, predictions):
    scored = frame.assign(_prediction=np.asarray(predictions, dtype=float))
    rows = []
    for site, group in scored.groupby('location_key', sort=True):
        metrics = _rank_metrics(group.yieldPerAcre, group._prediction, group.plot_key)
        by_capacity = {row['capacity']: row for row in metrics}
        rows.append({
            'site': site,
            'plots': len(group),
            'recall10': float(by_capacity[.1]['low_yield_recall']),
            'recall20': float(by_capacity[.2]['low_yield_recall']),
            'recall30': float(by_capacity[.3]['low_yield_recall']),
            'rank_mae20': float(by_capacity[.2]['rank_mae']),
        })
    return rows


def _add_spatial_image_context(data, index_columns):
    """Add unlabeled image summaries from orthogonal plot neighbors."""
    required = {'location_key', 'experiment_key', 'row', 'range'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Plot records lack spatial keys: {sorted(missing)}')
    coordinates = data[['row', 'range']].apply(pd.to_numeric, errors='coerce')
    values = coordinates.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.round(values)).all():
        raise ValueError('Plot row and range values must be finite integers.')
    if data.duplicated(['location_key', 'experiment_key', 'row', 'range']).any():
        raise ValueError('Spatial keys do not identify unique plots within experiments.')

    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    result = data.copy()
    neighbor_columns = []
    for source in index_columns:
        name = source.removeprefix('img_').removesuffix('_median')
        mean_column = f'img_{name}_neighbor_mean'
        delta_column = f'img_{name}_neighbor_delta'
        means = pd.Series(np.nan, index=result.index, dtype=float)
        counts = pd.Series(0, index=result.index, dtype=int)
        for _, group in result.groupby(['location_key', 'experiment_key'], sort=False):
            lookup = {
                (int(row), int(plot_range)): float(value)
                for row, plot_range, value in zip(
                    group.row, group['range'], group[source])
                if pd.notna(value)
            }
            for row_index, row, plot_range, own_value in zip(
                    group.index, group.row, group['range'], group[source]):
                if pd.isna(own_value):
                    continue
                neighbor_values = [
                    lookup[(int(row) + drow, int(plot_range) + drange)]
                    for drow, drange in offsets
                    if (int(row) + drow, int(plot_range) + drange) in lookup
                ]
                if neighbor_values:
                    means.at[row_index] = float(np.mean(neighbor_values))
                    counts.at[row_index] = len(neighbor_values)
        result[mean_column] = means
        result[delta_column] = result[source] - means
        neighbor_columns.extend((mean_column, delta_column))
    result['img_neighbor_count'] = counts
    neighbor_columns.append('img_neighbor_count')
    return result, neighbor_columns


def _add_temporal_trajectory_features(data, features_path, dap):
    """Add label-free index summaries from images available by each plot's DAP cutoff."""
    if dap is None:
        raise ValueError('Temporal trajectory features need a DAP cutoff.')
    features = pd.read_csv(features_path, parse_dates=['collection_date'])
    planting = data[['plot_key', 'plantingDate']].copy()
    planting['plantingDate'] = pd.to_datetime(
        planting['plantingDate'], format='mixed', errors='coerce')
    observations = features.merge(planting, on='plot_key', how='inner',
                                  validate='many_to_one')
    observations['observation_dap'] = (
        observations.collection_date - observations.plantingDate).dt.days
    observations = observations[
        observations.observation_dap.between(0, int(dap))].copy()
    summaries = []
    indices = ('ndvi', 'ndre', 'gndvi')
    for plot_key, group in observations.groupby('plot_key', sort=False):
        group = group.sort_values('observation_dap')
        row = {'plot_key': plot_key,
               'trajectory_observation_count': int(group.observation_dap.nunique()),
               'trajectory_span_dap': float(group.observation_dap.max()
                                            - group.observation_dap.min())}
        for index in indices:
            values = pd.to_numeric(group[f'img_{index}_median'], errors='coerce')
            valid = values.notna() & pd.to_numeric(
                group.observation_dap, errors='coerce').notna()
            x = group.loc[valid, 'observation_dap'].to_numpy(dtype=float)
            y = values.loc[valid].to_numpy(dtype=float)
            if len(y) == 0:
                row[f'trajectory_{index}_time_mean'] = np.nan
                row[f'trajectory_{index}_time_spread'] = np.nan
                row[f'trajectory_{index}_peak_dap'] = np.nan
            else:
                span = float(x.max() - x.min())
                time_mean = (float(np.trapezoid(y, x) / span) if span > 0
                             else float(y[-1]))
                row[f'trajectory_{index}_time_mean'] = time_mean
                row[f'trajectory_{index}_time_spread'] = float(np.std(y))
                row[f'trajectory_{index}_peak_dap'] = float(x[int(np.argmax(y))])
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    result = data.merge(summary, on='plot_key', how='left', validate='one_to_one')
    columns = [column for column in result
               if column.startswith('trajectory_')]
    return result, columns


def _fit_predict(estimator, model_name, train, target, predict_frame,
                 site_balanced_training=False):
    if site_balanced_training:
        site_sizes = train.groupby('location_key').location_key.transform('size')
        weights = len(train) / train.location_key.nunique() / site_sizes.to_numpy()
        estimator.fit(train, target, model__sample_weight=weights)
    else:
        estimator.fit(train, target)
    return estimator.predict(predict_frame)


def evaluate_rank_input_split(root, features, cutoff, output, seed=42,
                              include_catboost_native=False, dap=None,
                              cohort_dap=None,
                              include_management_interactions=False,
                              include_site_relative_indices=False,
                              site_balanced_training=False,
                              include_spatial_context=False,
                              include_temporal_trajectory=False,
                              selection_capacity=0.2):
    root, features, output = Path(root), Path(features), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder for every split run.')
    cutoff = None if dap is not None else pd.Timestamp(cutoff)
    if dap is None and cutoff.year != 2022:
        raise ValueError('This input comparison uses the 2022 development season.')
    if dap is not None and int(dap) < 0:
        raise ValueError('Days after planting must be zero or greater.')
    if float(selection_capacity) not in (.1, .2, .3):
        raise ValueError('Selection capacity must be 0.1, 0.2, or 0.3.')
    selection_capacity = float(selection_capacity)
    selection_recall_column = f'equal_site_recall{int(selection_capacity * 100)}'
    if cohort_dap is not None and dap is None:
        raise ValueError('A DAP cohort requires a DAP evaluation.')
    output.mkdir(parents=True, exist_ok=True)
    if not audit(root, output / 'source_audit.json')['structural_pass']:
        raise ValueError('Source audit failed. Review source_audit.json.')
    feature_audit = json.loads(features.with_suffix('.audit.json').read_text())
    if feature_audit['failed'] or feature_audit['settings']['season'] != 2022:
        raise ValueError('Feature audit failed or feature season does not match.')

    started = time.time()
    if dap is None:
        data, exclusions, image_cols = build_dataset(root, features, 2022, cutoff)
    else:
        data, exclusions, image_cols = build_dataset_by_dap(root, features, 2022, dap)
        if cohort_dap is not None:
            cohort, _, _ = build_dataset_by_dap(root, features, 2022, cohort_dap)
            required = set(cohort.plot_key)
            missing = required - set(data.plot_key)
            if missing:
                raise ValueError(f'{len(missing)} cohort plots lack images by DAP {dap}.')
            data = data[data.plot_key.isin(required)].copy()
    trajectory_cols = []
    if include_temporal_trajectory:
        data, trajectory_cols = _add_temporal_trajectory_features(
            data, features, dap)
    fields = ['poundsOfNitrogenPerAcre', 'irrigationProvided', 'planting_day']
    indices = [column for column in image_cols
               if column.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))
        or column in {'img_valid_fraction', 'img_observation_count',
                      'img_age_days', 'img_days_since_first',
                      'img_days_after_planting'}]
    relative_index_cols, image_context_cols = [], []
    if include_site_relative_indices:
        index_columns = [column for column in indices
                         if column.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))]
        missing = set(index_columns) - set(data.columns)
        if missing:
            raise ValueError(f'Image table lacks site-relative inputs: {sorted(missing)}')
        for column in index_columns:
            relative = f'{column}_within_site_pct'
            data[relative] = data.groupby('location_key')[column].rank(
                method='average', pct=True)
            relative_index_cols.append(relative)
        image_context_cols = [column for column in indices
                              if column not in index_columns]

    interaction_cols = []
    if include_management_interactions:
        management_features = (
            ('img_ndvi_median', 'ndvi'),
            ('img_ndre_median', 'ndre'),
            ('img_gndvi_median', 'gndvi'),
        )
        missing = {column for column, _ in management_features} - set(data.columns)
        if missing:
            raise ValueError(f'Image table lacks management interaction inputs: {sorted(missing)}')
        for source_column, name in management_features:
            nitrogen_column = f'img_{name}_x_nitrogen100'
            irrigation_column = f'img_{name}_x_irrigation'
            data[nitrogen_column] = (
                data[source_column] * data.poundsOfNitrogenPerAcre / 100.0)
            data[irrigation_column] = (
                data[source_column] * data.irrigationProvided)
            interaction_cols.extend((nitrogen_column, irrigation_column))

    spatial_cols = []
    if include_spatial_context:
        spatial_sources = [column for column in indices
                           if column in {'img_ndvi_median', 'img_ndre_median',
                                         'img_gndvi_median'}]
        if len(spatial_sources) != 3:
            raise ValueError('Spatial context needs NDVI, NDRE, and GNDVI medians.')
        data, spatial_cols = _add_spatial_image_context(data, spatial_sources)

    partitions, groups = randomized_trial_block_split(data, seed)
    manifest = data[['plot_key', 'location_key', 'experiment_key', 'block']].copy()
    manifest['trial_group'] = groups
    manifest['partition'] = partitions
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    parts = {name: data.iloc[np.flatnonzero(partitions == name)].copy()
             for name in ('train', 'validation', 'test')}
    train, validation, test = parts['train'], parts['validation'], parts['test']
    if validation.location_key.nunique() < 3:
        raise ValueError('Need at least three sites in validation data.')
    if test.location_key.nunique() != data.location_key.nunique():
        raise ValueError('The test split must include every site.')

    input_sets = {
        'nitrogen_plus_site': (['poundsOfNitrogenPerAcre'], ['location_key']),
        'nitrogen_indices_plus_site': (
            ['poundsOfNitrogenPerAcre'] + indices, ['location_key']),
        'fields_plus_site': (fields, ['genotype', 'location_key']),
        'indices_plus_site': (indices, ['location_key']),
        'fields_indices_plus_site': (fields + indices, ['genotype', 'location_key']),
    }
    if interaction_cols:
        input_sets['fields_indices_management_interactions_plus_site'] = (
            fields + indices + interaction_cols, ['genotype', 'location_key'])
    if relative_index_cols:
        input_sets['fields_relative_indices_plus_site'] = (
            fields + relative_index_cols + image_context_cols,
            ['genotype', 'location_key'])
    if spatial_cols:
        input_sets['fields_indices_spatial_context_plus_site'] = (
            fields + indices + spatial_cols, ['genotype', 'location_key'])
    if trajectory_cols:
        input_sets['fields_indices_temporal_trajectory_plus_site'] = (
            fields + indices + trajectory_cols, ['genotype', 'location_key'])
    cores = os.cpu_count() or 1
    validation_site_rows, candidate_rows, selected_rows = [], [], []
    validation_candidate_predictions = {}
    selected_by_input, candidates_by_input = {}, {}
    for input_name, (numeric, categorical) in input_sets.items():
        make_candidates = lambda model_seed: _candidates(
            numeric, categorical, cores, model_seed,
            include_catboost_native=include_catboost_native)
        candidate_names = [name for name, _ in make_candidates(seed)]
        candidates_by_input[input_name] = candidate_names
        scores = []
        for candidate_name in candidate_names:
            seed_predictions = []
            for model_seed in MODEL_SEEDS:
                estimator = dict(make_candidates(model_seed))[candidate_name]
                seed_predictions.append(_fit_predict(
                    estimator, candidate_name, train,
                    _site_percentile(train).to_numpy(), validation,
                    site_balanced_training=site_balanced_training))
            predicted = np.mean(seed_predictions, axis=0)
            validation_candidate_predictions[(input_name, candidate_name)] = predicted
            site_rows = _site_scores(validation, predicted)
            validation_site_rows.extend({
                'split_seed': seed,
                'input_set': input_name,
                'estimator': candidate_name,
                **row,
            } for row in site_rows)
            scores.append({
                'input_set': input_name,
                'estimator': candidate_name,
                'equal_site_recall10': float(np.mean([r['recall10'] for r in site_rows])),
                'equal_site_recall20': float(np.mean([r['recall20'] for r in site_rows])),
                'equal_site_recall30': float(np.mean([r['recall30'] for r in site_rows])),
                'equal_site_rank_mae20': float(np.mean([r['rank_mae20'] for r in site_rows])),
                'validation_sites': len(site_rows),
            })
        chosen = min(scores, key=lambda row: (
            -row[selection_recall_column], row['equal_site_rank_mae20'], row['estimator']))
        selected_by_input[input_name] = chosen['estimator']
        candidate_rows.extend(scores)
        selected_rows.append(chosen)

    selected_input = min(selected_rows, key=lambda row: (
        -row[selection_recall_column], row['equal_site_rank_mae20'], row['input_set']))['input_set']
    validation_prediction_frame = validation[[
        'plot_key', 'location_key', 'experiment_key', 'block', 'genotype',
        'collection_date', 'yieldPerAcre']].copy()
    validation_prediction_frame = validation_prediction_frame.rename(
        columns={'yieldPerAcre': 'actual_yield'})
    validation_prediction_frame['split_seed'] = seed
    for input_name, selected_model in selected_by_input.items():
        validation_prediction_frame[f'{input_name}_seed_mean'] = (
            validation_candidate_predictions[(input_name, selected_model)])
        validation_prediction_frame[f'{input_name}_selected_model'] = selected_model
    validation_prediction_frame['validation_selected_input'] = selected_input
    validation_prediction_frame.to_csv(
        output / 'validation_predictions.csv', index=False)
    development = pd.concat([train, validation], ignore_index=True)
    development_target = _site_percentile(development).to_numpy()
    prediction_frame = test[['plot_key', 'location_key', 'experiment_key', 'block',
                             'genotype', 'collection_date', 'yieldPerAcre']].copy()
    prediction_frame = prediction_frame.rename(columns={'yieldPerAcre': 'actual_yield'})
    prediction_frame['split_seed'] = seed
    metric_rows = []
    for input_name, (numeric, categorical) in input_sets.items():
        chosen = selected_by_input[input_name]
        make_candidates = lambda model_seed: _candidates(
            numeric, categorical, cores, model_seed,
            include_catboost_native=include_catboost_native)
        seed_predictions = {}
        for model_seed in MODEL_SEEDS:
            estimator = dict(make_candidates(model_seed))[chosen]
            seed_predictions[model_seed] = _fit_predict(
                estimator, chosen, development, development_target, test,
                site_balanced_training=site_balanced_training)
            prediction_frame[f'{input_name}_seed_{model_seed}'] = seed_predictions[model_seed]
        prediction_frame[f'{input_name}_seed_mean'] = np.mean(
            [seed_predictions[value] for value in MODEL_SEEDS], axis=0)
        prediction_frame[f'{input_name}_selected_model'] = chosen
        for site, group in prediction_frame.groupby('location_key', sort=True):
            for score in _rank_metrics(group.actual_yield,
                                       group[f'{input_name}_seed_mean'],
                                       group.plot_key):
                metric_rows.append({
                    'split_seed': seed,
                    'input_set': input_name,
                    'selected_estimator': chosen,
                    'site': site,
                    **score,
                })
        prediction_frame.to_csv(output / 'test_predictions.partial.csv', index=False)
        validation_recall = next(
            row['equal_site_recall20'] for row in selected_rows
            if row['input_set'] == input_name)
        print(f'{input_name}: selected {chosen}; validation recall '
              f'{validation_recall:.3f}', flush=True)
    prediction_frame['validation_selected_input'] = selected_input
    prediction_frame.to_csv(output / 'test_predictions.csv', index=False)
    (output / 'test_predictions.partial.csv').unlink(missing_ok=True)

    metrics = pd.DataFrame(metric_rows)
    aggregate = metrics.groupby(['input_set', 'selected_estimator', 'capacity']).agg(
        equal_site_recall=('low_yield_recall', 'mean'),
        site_sd=('low_yield_recall', 'std'),
        equal_site_precision=('low_yield_precision', 'mean'),
        random_recall=('random_recall', 'mean'),
        sites=('site', 'nunique'),
    ).reset_index()
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame['selected_for_input'] = candidate_frame.apply(
        lambda row: row.estimator == selected_by_input[row.input_set], axis=1)
    selected_frame = candidate_frame[candidate_frame.selected_for_input].sort_values(
        ['equal_site_recall20', 'equal_site_rank_mae20', 'input_set'],
        ascending=[False, True, True])
    metrics.to_csv(output / 'test_metrics_by_site.csv', index=False)
    aggregate.to_csv(output / 'test_aggregate_metrics.csv', index=False)
    pd.DataFrame(validation_site_rows).to_csv(
        output / 'validation_scores_by_site.csv', index=False)
    candidate_frame.to_csv(output / 'validation_candidate_scores.csv', index=False)
    selected_frame.to_csv(output / 'validation_input_scores.csv', index=False)
    run = {
        'schema_version': 1,
        'season': 2022,
        'cutoff': cutoff.date().isoformat() if cutoff is not None else None,
        'days_after_planting': int(dap) if dap is not None else None,
        'cohort_dap': int(cohort_dap) if cohort_dap is not None else None,
        'target': 'within-block lowest-yield plot recall at 20% crew capacity',
        'split_strategy': 'randomized, site-stratified, whole trial blocks',
        'split_seed': seed,
        'validation_selection': (
            f'equal-site recall at {selection_capacity:.0%}; rank MAE at 20% breaks ties'),
        'validation_selection_capacity': selection_capacity,
        'test_used_for_selection': False,
        'split_rules': {
            'sites_with_three_or_more_blocks': 'one validation block, one test block, remaining blocks for training',
            'sites_with_two_blocks': 'one training block and one test block; no validation block',
            'refit': 'selected input and model refit on training and validation blocks before test scoring',
        },
        'selected_input': selected_input,
        'selected_models': selected_by_input,
        'validation_scores': selected_rows,
        'input_sets': list(input_sets),
        'include_management_interactions': include_management_interactions,
        'management_interaction_columns': interaction_cols,
        'include_site_relative_indices': include_site_relative_indices,
        'include_spatial_context': include_spatial_context,
        'spatial_context_columns': spatial_cols,
        'spatial_context_method': (
            'mean and plot-minus-mean for orthogonal neighbors in the same site and experiment; image values only'
            if include_spatial_context else None),
        'spatial_context_uses_unlabeled_images_across_splits': include_spatial_context,
        'include_temporal_trajectory': include_temporal_trajectory,
        'temporal_trajectory_columns': trajectory_cols,
        'temporal_trajectory_method': (
            'per-plot time-weighted mean, temporal spread, peak DAP, observation count, '
            'and observation span from images at or before the DAP cutoff; no yield labels'
            if include_temporal_trajectory else None),
        'site_balanced_training': site_balanced_training,
        'site_relative_index_columns': relative_index_cols,
        'site_relative_transform': (
            'within-site ranks from all eligible plot imagery without yield labels'
            if include_site_relative_indices else None),
        'candidate_models': candidates_by_input,
        'include_catboost_native': include_catboost_native,
        'prediction_seeds': list(MODEL_SEEDS),
        'validation_sites': sorted(validation.location_key.unique()),
        'test_sites': sorted(test.location_key.unique()),
        'eligible_plots': len(data),
        'test_plots': len(test),
        'group_count': len(np.unique(groups)),
        'source_hash': _hash(features),
        'plot_table_hash': _hash(root / 'data/processed/plot_records.csv'),
        'code_revision': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'evaluation_source_hashes': {
            relative: _hash(root / relative) for relative in CODE_FILES
        },
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'elapsed_seconds': round(time.time() - started, 1),
        'limitations': (
            'Early DAP images omit MOValley. This cohort uses four sites and one season.'
            if 'MOValley' not in set(data.location_key)
            else 'MOValley has two blocks, so validation omits it. This split tests one '
                 'block per site. Results use one season.'),
    }
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(f'Saved rank input comparison to {output}', flush=True)
