"""Compare yield models on separate randomized trial-block train, validation, and test data."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .audit import audit
from .dataset import build_dataset, build_dataset_by_dap
from .model import _candidates
from .metrics import score_predictions
from .random_eval import _add_index_trends, _fit_predict
from .splits import randomized_trial_block_split


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _site_validation_scores(frame, prediction):
    scored = frame.assign(_prediction=np.asarray(prediction, dtype=float))
    rows = []
    for site, group in scored.groupby('location_key', sort=True):
        actual = group.yieldPerAcre.to_numpy(float)
        predicted = group._prediction.to_numpy(float)
        recall = next(row['low_yield_recall'] for row in score_predictions(
            actual, predicted, group.plot_key, site, 'validation')
            [0] if row['capacity'] == .2)
        rows.append({
            'site': site,
            'plots': len(group),
            'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
            'recall20': float(recall),
        })
    return rows


def _add_dap_index_trends(data, features_path, dap):
    """Add per-plot vegetation-index slopes per ten DAP using prior images only."""
    source = pd.read_csv(features_path, dtype={'plot_key': str},
                         parse_dates=['collection_date'])
    planting = data[['plot_key', 'plantingDate']].copy()
    planting['plot_key'] = planting.plot_key.astype(str)
    source['plot_key'] = source.plot_key.astype(str)
    source = source.merge(planting, on='plot_key', validate='many_to_one')
    source['image_dap'] = (source.collection_date - source.plantingDate).dt.days
    source = source[source.image_dap.between(0, int(dap))]
    source_columns = ('img_ndvi_median', 'img_ndre_median', 'img_gndvi_median')
    missing = set(source_columns).difference(source.columns)
    if missing:
        raise ValueError(f'Image table lacks index fields for DAP trends: {sorted(missing)}')

    trends = {}
    for source_column in source_columns:
        target_column = source_column.replace('_median', '_slope_dap10d')
        slopes = {}
        for plot_key, group in source.groupby('plot_key', sort=False):
            x = group.image_dap.to_numpy(float) / 10.0
            y = pd.to_numeric(group[source_column], errors='coerce').to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            slopes[plot_key] = (
                float(np.polyfit(x, y, 1)[0])
                if len(x) >= 2 and np.ptp(x) > 0 else np.nan
            )
        trends[target_column] = pd.Series(slopes, name=target_column)
    trend_frame = pd.DataFrame(trends).rename_axis('plot_key').reset_index()
    result = data.copy()
    result['plot_key'] = result.plot_key.astype(str)
    result = result.merge(trend_frame, on='plot_key', validate='one_to_one')
    if len(result) != len(data):
        raise ValueError('DAP trend feature join changed the plot count.')
    return result, list(trends)


def evaluate_yield_split(root, features, cutoff, output, seed=42,
                         include_catboost_native=False,
                         include_image_trends=False,
                         include_catboost_mae=False, dap=None,
                         cohort_dap=None, include_dap_trends=False,
                         include_raw_band_input=False,
                         include_management_interactions=False):
    """Choose yield models on validation, refit on train plus validation, and score test blocks."""
    root, features, output = Path(root), Path(features), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder for every split run.')
    cutoff = None if dap is not None else pd.Timestamp(cutoff)
    if dap is None and cutoff.year != 2022:
        raise ValueError('The randomized split check uses the 2022 development season.')
    if dap is not None and int(dap) < 0:
        raise ValueError('Days after planting must be zero or greater.')
    if cohort_dap is not None and dap is None:
        raise ValueError('A DAP cohort requires a DAP evaluation.')
    if dap is not None and include_image_trends:
        raise ValueError('Image trend ablations use calendar-date cutoffs.')
    if dap is None and include_dap_trends:
        raise ValueError('DAP trend features require a DAP cutoff.')
    output.mkdir(parents=True, exist_ok=True)
    if not audit(root, output / 'source_audit.json')['structural_pass']:
        raise ValueError('The source audit failed. Review source_audit.json.')
    feature_audit = json.loads(features.with_suffix('.audit.json').read_text())
    if feature_audit['failed'] or feature_audit['settings']['season'] != 2022:
        raise ValueError('Feature audit failed or feature season does not match.')

    started = time.time()
    if dap is None:
        data, exclusions, image_cols = build_dataset(root, features, 2022, cutoff)
    else:
        data, exclusions, image_cols = build_dataset_by_dap(root, features, 2022, dap)
        dap_trend_cols = []
        if include_dap_trends:
            data, dap_trend_cols = _add_dap_index_trends(data, features, dap)
        if cohort_dap is not None:
            cohort, _, _ = build_dataset_by_dap(root, features, 2022, cohort_dap)
            required = set(cohort.plot_key)
            missing = required - set(data.plot_key)
            if missing:
                raise ValueError(f'{len(missing)} cohort plots lack images by DAP {dap}.')
            data = data[data.plot_key.isin(required)].copy()
    if dap is None:
        dap_trend_cols = []
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
    trend_cols = []
    if include_image_trends:
        data, trend_cols = _add_index_trends(data, features, cutoff)
    partitions, groups = randomized_trial_block_split(data, seed)
    manifest = data[['plot_key', 'location_key', 'experiment_key', 'block']].copy()
    manifest['trial_group'] = groups
    manifest['partition'] = partitions
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    manifest.to_csv(output / 'split_manifest.csv', index=False)
    split_data = {name: data.iloc[np.flatnonzero(partitions == name)].copy()
                  for name in ('train', 'validation', 'test')}
    train, validation, test = (split_data['train'], split_data['validation'],
                               split_data['test'])
    if validation.location_key.nunique() < 3:
        raise ValueError('Need at least three sites in validation data.')
    if test.location_key.nunique() != data.location_key.nunique():
        raise ValueError('The test split must include every site.')

    field_numeric = ['poundsOfNitrogenPerAcre', 'irrigationProvided', 'planting_day']
    indices = [column for column in image_cols
               if column.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))
        or column in {'img_valid_fraction', 'img_observation_count',
                      'img_age_days', 'img_days_since_first',
                      'img_days_after_planting'}]
    input_sets = {
        'nitrogen_plus_site': (['poundsOfNitrogenPerAcre'], ['location_key']),
        'fields_plus_site': (field_numeric, ['genotype', 'location_key']),
        'images_plus_site': (image_cols, ['location_key']),
        'indices_plus_site': (indices, ['location_key']),
        'fields_indices_plus_site': (field_numeric + indices,
                                    ['genotype', 'location_key']),
    }
    if dap_trend_cols:
        input_sets['indices_dap_trend_plus_site'] = (
            indices + dap_trend_cols, ['location_key'])
        input_sets['fields_indices_dap_trend_plus_site'] = (
            field_numeric + indices + dap_trend_cols,
            ['genotype', 'location_key'])
    if include_raw_band_input:
        input_sets['fields_images_plus_site'] = (
            field_numeric + image_cols, ['genotype', 'location_key'])
    if interaction_cols:
        input_sets['fields_indices_management_interactions_plus_site'] = (
            field_numeric + indices + interaction_cols,
            ['genotype', 'location_key'])
    if include_image_trends:
        input_sets['indices_trend_plus_site'] = (
            indices + trend_cols, ['location_key'])
        input_sets['fields_indices_trend_plus_site'] = (
            field_numeric + indices + trend_cols, ['genotype', 'location_key'])
    cores = os.cpu_count() or 1
    model_seeds = (42, 101, 2026)
    validation_rows = []
    validation_summary = []
    selected_by_input = {}
    validation_prediction_by_input = {}
    predictions = test[['plot_key', 'location_key', 'experiment_key', 'block',
                        'genotype', 'collection_date', 'yieldPerAcre']].copy()
    predictions = predictions.rename(columns={'yieldPerAcre': 'actual_yield'})
    development = pd.concat([train, validation], ignore_index=True)
    global_mean = float(development.yieldPerAcre.mean())
    site_means = development.groupby('location_key').yieldPerAcre.mean()
    predictions['training_mean'] = global_mean
    predictions['training_site_mean'] = predictions.location_key.map(site_means).fillna(global_mean)
    candidate_names_by_input = {}

    for input_set, (numeric, categorical) in input_sets.items():
        candidate_map = dict(_candidates(
            numeric, categorical, cores, seed,
            include_catboost_native=include_catboost_native,
            include_catboost_mae=include_catboost_mae))
        candidate_names_by_input[input_set] = list(candidate_map)
        candidate_scores = []
        candidate_validation_predictions = {}
        for candidate_name in candidate_map:
            validation_predictions = []
            for model_seed in model_seeds:
                estimator = dict(_candidates(
                    numeric, categorical, cores, model_seed,
                    include_catboost_native=include_catboost_native,
                    include_catboost_mae=include_catboost_mae))[candidate_name]
                validation_predictions.append(_fit_predict(
                    estimator, candidate_name, train,
                    train.yieldPerAcre.to_numpy(), validation))
            predicted = np.mean(validation_predictions, axis=0)
            candidate_validation_predictions[candidate_name] = predicted
            site_scores = _site_validation_scores(validation, predicted)
            for row in site_scores:
                validation_rows.append({
                    'split_seed': seed, 'input_set': input_set,
                    'estimator': candidate_name, **row,
                })
            candidate_scores.append({
                'input_set': input_set,
                'estimator': candidate_name,
                'equal_site_mae': float(np.mean([row['mae'] for row in site_scores])),
                'equal_site_rmse': float(np.mean([row['rmse'] for row in site_scores])),
                'equal_site_recall20': float(np.mean([row['recall20'] for row in site_scores])),
                'validation_sites': len(site_scores),
            })
            pd.DataFrame(validation_rows).to_csv(
                output / 'validation_scores_by_site.partial.csv', index=False)
        chosen = min((row for row in candidate_scores), key=lambda row: (
            row['equal_site_mae'], row['equal_site_rmse'], row['estimator']))
        selected_by_input[input_set] = chosen['estimator']
        validation_prediction_by_input[input_set] = candidate_validation_predictions[
            chosen['estimator']]
        validation_summary.extend(candidate_scores)

        test_seed_predictions = {}
        for model_seed in model_seeds:
            estimator = dict(_candidates(
                numeric, categorical, cores, model_seed,
                include_catboost_native=include_catboost_native,
                include_catboost_mae=include_catboost_mae))[chosen['estimator']]
            test_seed_predictions[model_seed] = _fit_predict(
                estimator, chosen['estimator'], development,
                development.yieldPerAcre.to_numpy(), test)
            predictions[f'{input_set}_seed_{model_seed}'] = test_seed_predictions[model_seed]
        predictions[f'{input_set}_seed_mean'] = np.mean(
            [test_seed_predictions[s] for s in model_seeds], axis=0)
        predictions[f'{input_set}_selected_model'] = chosen['estimator']
        predictions.to_csv(output / 'test_predictions.partial.csv', index=False)
        print(f'{input_set}: selected {chosen["estimator"]}; validation MAE '
              f'{chosen["equal_site_mae"]:.2f}', flush=True)

    selected_input = min(input_sets, key=lambda name: (
        min(row['equal_site_mae'] for row in validation_summary
            if row['input_set'] == name),
        min(row['equal_site_rmse'] for row in validation_summary
            if row['input_set'] == name), name))
    validation_prediction_frame = validation[[
        'plot_key', 'location_key', 'experiment_key', 'block', 'genotype',
        'collection_date', 'yieldPerAcre']].copy()
    validation_prediction_frame = validation_prediction_frame.rename(
        columns={'yieldPerAcre': 'actual_yield'})
    validation_prediction_frame['split_seed'] = seed
    for input_set, predicted in validation_prediction_by_input.items():
        validation_prediction_frame[f'{input_set}_seed_mean'] = predicted
        validation_prediction_frame[f'{input_set}_selected_model'] = selected_by_input[input_set]
    validation_prediction_frame['validation_selected_input'] = selected_input
    validation_prediction_frame.to_csv(
        output / 'validation_predictions.csv', index=False)
    predictions['validation_selected_input'] = selected_input
    predictions.to_csv(output / 'test_predictions.csv', index=False)

    score_columns = {
        'training_mean': 'training_mean',
        'training_site_mean': 'training_site_mean',
    }
    for input_set in input_sets:
        score_columns[input_set] = f'{input_set}_seed_mean'
    metric_rows = []
    for site, group in predictions.groupby('location_key', sort=True):
        for model_name, column in score_columns.items():
            scored, _ = score_predictions(
                group.actual_yield, group[column], group.plot_key,
                site, model_name)
            metric_rows.extend({'split_seed': seed, 'site': site,
                                'selected_estimator': selected_by_input.get(model_name, model_name),
                                **row} for row in scored)
    metrics = pd.DataFrame(metric_rows)
    aggregate = metrics.groupby(['split_seed', 'model', 'capacity']).apply(
        lambda part: pd.Series({
            'equal_site_mae': part.drop_duplicates('site').mae.mean(),
            'equal_site_rmse': part.drop_duplicates('site').rmse.mean(),
            'plot_weighted_mae': np.average(
                part.drop_duplicates('site').mae,
                weights=part.drop_duplicates('site').n),
            'plot_weighted_rmse': np.average(
                part.drop_duplicates('site').rmse,
                weights=part.drop_duplicates('site').n),
            'equal_site_recall': part.low_yield_recall.mean(),
            'equal_site_precision': part.low_yield_precision.mean(),
            'random_recall': part.random_recall.mean(),
            'sites': part.site.nunique(),
        }), include_groups=False).reset_index()
    candidate_frame = pd.DataFrame(validation_summary)
    candidate_frame['selected_for_input'] = candidate_frame.apply(
        lambda row: row.estimator == selected_by_input[row.input_set], axis=1)
    input_summary = candidate_frame[candidate_frame.selected_for_input].sort_values(
        ['equal_site_mae', 'equal_site_rmse', 'input_set'])

    partition_counts = manifest.groupby(['partition', 'location_key']).agg(
        plots=('plot_key', 'size'), blocks=('trial_group', 'nunique')).reset_index()
    run = {
        'schema_version': 1,
        'season': 2022,
        'cutoff': str(cutoff.date()) if cutoff is not None else None,
        'days_after_planting': int(dap) if dap is not None else None,
        'cohort_dap': int(cohort_dap) if cohort_dap is not None else None,
        'target': 'final yield per acre in source units',
        'split_strategy': 'randomized, site-stratified trial-block train/validation/test',
        'split_seed': seed,
        'split_rules': {
            'sites_with_three_or_more_blocks': 'one validation block, one test block, remaining blocks for training',
            'sites_with_two_blocks': 'one training block and one test block; no validation block',
            'refit': 'selected input and model refit on training and validation blocks before test scoring',
            'test_used_for_selection': False,
        },
        'validation_sites': sorted(validation.location_key.unique()),
        'test_sites': sorted(test.location_key.unique()),
        'partition_counts': partition_counts.to_dict(orient='records'),
        'validation_selection': 'equal-site mean absolute error',
        'selected_input': selected_input,
        'selected_models': selected_by_input,
        'input_sets': list(input_sets),
        'candidate_models': candidate_names_by_input,
        'include_catboost_native': include_catboost_native,
        'include_catboost_mae': include_catboost_mae,
        'include_image_trends': include_image_trends,
        'include_dap_trends': include_dap_trends,
        'include_raw_band_input': include_raw_band_input,
        'include_management_interactions': include_management_interactions,
        'management_interaction_columns': interaction_cols,
        'trend_columns': trend_cols,
        'dap_trend_columns': dap_trend_cols,
        'prediction_seeds': list(model_seeds),
        'eligible_plots': len(data),
        'test_plots': len(test),
        'group_count': len(np.unique(groups)),
        'limitations': (
            'Early DAP images omit MOValley. This cohort uses four sites and one season.'
            if 'MOValley' not in set(data.location_key)
            else 'MOValley has two trial blocks. One stays in training and one stays in test, '
                 'so validation omits MOValley. The test score uses one block per site. '
                 'The data covers one season and five sites.'),
        'source_hash': _hash(features),
        'plot_table_hash': _hash(root / 'data/processed/plot_records.csv'),
        'code_revision': subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'elapsed_seconds': round(time.time() - started, 1),
    }
    metrics.to_csv(output / 'test_metrics_by_site.csv', index=False)
    aggregate.to_csv(output / 'test_aggregate_metrics.csv', index=False)
    candidate_frame.to_csv(output / 'validation_candidate_scores.csv', index=False)
    input_summary.to_csv(output / 'validation_input_scores.csv', index=False)
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(f'Saved train/validation/test yield evaluation to {output}', flush=True)
