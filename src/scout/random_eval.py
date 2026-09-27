"""Nested randomized trial-block validation for known-site scouting."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from .audit import audit
from .dataset import build_dataset
from .metrics import score_predictions
from .model import _candidates, _prepare
from .splits import _trial_groups


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _macro_site_mae(actual, predicted, sites):
    frame = pd.DataFrame({'actual': actual, 'predicted': predicted, 'site': sites})
    return float(frame.groupby('site').apply(
        lambda part: mean_absolute_error(part.actual, part.predicted), include_groups=False
    ).mean())


def _add_index_trends(data, features_path, cutoff):
    """Add date-safe, per-plot index slopes, scaled as change per ten days."""
    source = pd.read_csv(features_path, parse_dates=['collection_date'])
    source = source[source.collection_date <= cutoff].copy()
    source['plot_key'] = source.plot_key.astype(str)
    source = source[source.plot_key.isin(data.plot_key.astype(str))]
    trend_columns = ['img_ndvi_median', 'img_ndre_median', 'img_gndvi_median']
    missing = set(trend_columns).difference(source.columns)
    if missing:
        raise ValueError(f'Image table lacks index fields for trend features: {sorted(missing)}')

    trends = {}
    for source_column in trend_columns:
        target_column = source_column.replace('_median', '_slope_10d')
        slope_values = {}
        for plot_key, group in source.groupby('plot_key', sort=False):
            days = (group.collection_date - group.collection_date.min()).dt.days.to_numpy(float)
            values = pd.to_numeric(group[source_column], errors='coerce').to_numpy(float)
            valid = np.isfinite(days) & np.isfinite(values)
            x = days[valid] / 10.0
            y = values[valid]
            if len(x) >= 2 and np.ptp(x) > 0:
                slope_values[plot_key] = float(np.polyfit(x, y, 1)[0])
            else:
                slope_values[plot_key] = np.nan
        trends[target_column] = pd.Series(slope_values, name=target_column)

    trend_frame = pd.DataFrame(trends).rename_axis('plot_key').reset_index()
    result = data.copy()
    result['plot_key'] = result.plot_key.astype(str)
    result = result.merge(trend_frame, on='plot_key', validate='one_to_one')
    if len(result) != len(data):
        raise ValueError('Trend feature join changed the plot count.')
    return result, list(trends)


def _fit_predict(estimator, model_name, train, target, predict_frame):
    if model_name == 'hist_gradient_leaf15':
        # sklearn 1.9 can emit this worker warning once per thread task during
        # histogram binning. This run uses no non-default sklearn config.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='`sklearn.utils.parallel.delayed` should be used with',
                category=UserWarning,
                module='sklearn.utils.parallel',
            )
            estimator.fit(train, target)
            return estimator.predict(predict_frame)
    estimator.fit(train, target)
    return estimator.predict(predict_frame)


def evaluate_randomized(root, features, cutoff, output, seed=42, outer_folds=5,
                        inner_folds=4, objective='macro_mae', include_image_trends=False,
                        include_catboost_mae=False):
    """Run nested CV with site/experiment/block groups held intact."""
    root, features, output = Path(root), Path(features), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder for every model run.')
    output.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.year != 2022:
        raise ValueError('The current randomized evaluation supports the 2022 development set.')
    if objective not in {'macro_mae', 'recall20'}:
        raise ValueError("objective must be 'macro_mae' or 'recall20'.")
    if not audit(root, output / 'source_audit.json')['structural_pass']:
        raise ValueError('The source audit failed. Review the data audit before evaluation.')
    feature_audit = json.loads(features.with_suffix('.audit.json').read_text())
    if feature_audit['failed'] or feature_audit['settings']['season'] != 2022:
        raise ValueError('Feature audit failed or feature season does not match.')

    started = time.time()
    data, exclusions, image_cols = build_dataset(root, features, 2022, cutoff)
    trend_cols = []
    if include_image_trends:
        data, trend_cols = _add_index_trends(data, features, cutoff)
    group_ids = _trial_groups(data)
    group_count = len(np.unique(group_ids))
    if group_count < outer_folds:
        raise ValueError(f'Need {outer_folds} trial blocks; found {group_count}.')
    cores = os.cpu_count() or 1
    field_numeric = ['poundsOfNitrogenPerAcre', 'irrigationProvided', 'planting_day']
    indices = [c for c in image_cols if c.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))
               or c in {'img_valid_fraction', 'img_observation_count', 'img_age_days', 'img_days_since_first'}]
    index_base_cols = [column for column in indices if column not in trend_cols]
    # Site is known when the crew works in a trial. Include it in every model
    # so the image comparison does not confuse site differences with image gain.
    groups = {
        'nitrogen_plus_site': (['poundsOfNitrogenPerAcre'], ['location_key']),
        'fields_plus_site': (field_numeric, ['genotype', 'location_key']),
        'raw_images_plus_site': (image_cols, ['location_key']),
        'indices_plus_site': (index_base_cols, ['location_key']),
        'fields_raw_images_plus_site': (field_numeric + image_cols, ['genotype', 'location_key']),
        'fields_indices_plus_site': (field_numeric + index_base_cols, ['genotype', 'location_key']),
    }
    if include_image_trends:
        groups['fields_indices_trend_plus_site'] = (
            field_numeric + index_base_cols + trend_cols, ['genotype', 'location_key'])
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    data[['plot_key', 'location_key', 'experiment_key', 'block', 'collection_date']].to_csv(
        output / 'eligible_plots.csv', index=False)

    outer = GroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    held_predictions, selection_records = [], []
    y = data.yieldPerAcre.to_numpy()
    sites = data.location_key.to_numpy()
    for fold_num, (train_idx, test_idx) in enumerate(outer.split(data, y, group_ids), 1):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        train_y = train.yieldPerAcre.to_numpy()
        train_groups = _trial_groups(train)
        inner_count = min(inner_folds, len(np.unique(train_groups)))
        inner = GroupKFold(n_splits=inner_count, shuffle=True, random_state=seed + fold_num)
        print(f'Outer randomized block fold {fold_num}/{outer_folds}: '
              f'{len(train)} train, {len(test)} test plots, {inner_count} inner folds', flush=True)

        for baseline in ('training_mean', 'training_site_mean'):
            pred = (np.full(len(test), float(np.mean(train_y))) if baseline == 'training_mean'
                    else test.location_key.map(train.groupby('location_key').yieldPerAcre.mean())
                    .fillna(float(np.mean(train_y))).to_numpy())
            held_predictions.append(pd.DataFrame({
                'plot_key': test.plot_key.astype(str).to_numpy(), 'location_key': test.location_key.to_numpy(),
                'experiment_key': test.experiment_key.to_numpy(), 'block': test.block.to_numpy(),
                'fold': fold_num, 'input_set': baseline, 'estimator': baseline,
                'actual_yield': test.yieldPerAcre.to_numpy(), 'pred_seed_42': pred,
                'pred_seed_101': pred, 'pred_seed_2026': pred,
            }))

        for input_set, (numeric, categorical) in groups.items():
            candidate_scores, candidate_recalls = {}, {}
            for candidate_name, _ in _candidates(
                    numeric, categorical, cores, seed,
                    include_catboost_mae=include_catboost_mae):
                fold_scores, fold_recalls = [], []
                for inner_fold, (inner_train_idx, valid_idx) in enumerate(
                        inner.split(train, train_y, train_groups), 1):
                    inner_train, valid = train.iloc[inner_train_idx], train.iloc[valid_idx]
                    estimator = dict(_candidates(
                        numeric, categorical, cores, seed,
                        include_catboost_mae=include_catboost_mae))[candidate_name]
                    valid_pred = _fit_predict(estimator, candidate_name, inner_train,
                                              inner_train.yieldPerAcre.to_numpy(), valid)
                    fold_scores.append(_macro_site_mae(
                        valid.yieldPerAcre.to_numpy(), valid_pred, valid.location_key.to_numpy()))
                    site_recalls = []
                    for valid_site, site_rows in valid.assign(_predicted=valid_pred).groupby('location_key'):
                        site_score, _ = score_predictions(site_rows.yieldPerAcre,
                                                          site_rows['_predicted'],
                                                          site_rows.plot_key,
                                                          valid_site, candidate_name)
                        fold_score = next(row for row in site_score if row['capacity'] == .2)
                        site_recalls.append(fold_score['low_yield_recall'])
                    fold_recalls.append(float(np.mean(site_recalls)))
                    selection_records.append({'outer_fold': fold_num, 'inner_fold': inner_fold,
                                              'input_set': input_set, 'estimator': candidate_name,
                                              'macro_site_mae': fold_scores[-1],
                                              'macro_site_recall20': fold_recalls[-1],
                                              'validation_plots': len(valid),
                                              'validation_blocks': len(np.unique(train_groups[valid_idx]))})
                candidate_scores[candidate_name] = float(np.mean(fold_scores))
                candidate_recalls[candidate_name] = float(np.mean(fold_recalls))
            if objective == 'macro_mae':
                chosen = min(candidate_scores, key=lambda name: (candidate_scores[name], name))
            else:
                chosen = min(candidate_recalls, key=lambda name: (-candidate_recalls[name], name))
            selected_name = chosen
            seed_predictions = {}
            for model_seed in (42, 101, 2026):
                estimator = dict(_candidates(
                    numeric, categorical, cores, model_seed,
                    include_catboost_mae=include_catboost_mae))[chosen]
                seed_predictions[model_seed] = _fit_predict(
                    estimator, chosen, train, train_y, test)
            held_predictions.append(pd.DataFrame({
                'plot_key': test.plot_key.astype(str).to_numpy(), 'location_key': test.location_key.to_numpy(),
                'experiment_key': test.experiment_key.to_numpy(), 'block': test.block.to_numpy(),
                'fold': fold_num, 'input_set': input_set, 'estimator': selected_name,
                'actual_yield': test.yieldPerAcre.to_numpy(),
                'pred_seed_42': seed_predictions[42],
                'pred_seed_101': seed_predictions[101],
                'pred_seed_2026': seed_predictions[2026],
            }))
            # Save progress after each model family so a stopped run keeps its evidence.
            pd.DataFrame(selection_records).to_csv(output / 'inner_selection_scores.partial.csv', index=False)
            pd.concat(held_predictions, ignore_index=True).to_csv(
                output / 'predictions.partial.csv', index=False)
            print(f'  {input_set}: selected {chosen}; inner macro MAE '
                  f'{candidate_scores[chosen]:.2f} bu/acre; recall20 '
                  f'{candidate_recalls[chosen]:.3f}', flush=True)

    predictions = pd.concat(held_predictions, ignore_index=True)
    if predictions.duplicated(['plot_key', 'input_set']).any():
        raise ValueError('Each plot must have one out-of-fold prediction per input set.')
    details = data[['plot_key', 'genotype', 'row', 'range', 'collection_date',
                    'img_age_days', 'img_observation_count']].copy()
    predictions = predictions.merge(details, on='plot_key', validate='many_to_one')
    metrics, stability, app_rows = [], [], []
    for (input_set, site), part in predictions.groupby(['input_set', 'location_key']):
        seeds = ['pred_seed_42', 'pred_seed_101', 'pred_seed_2026']
        seed_sets = {}
        seed_ranks = {}
        for column in seeds:
            scores, order = score_predictions(part.actual_yield, part[column], part.plot_key,
                                              site, input_set)
            if column == 'pred_seed_42':
                metrics.extend(scores)
            rank = np.empty(len(order), dtype=int)
            rank[order] = np.arange(1, len(order) + 1)
            seed_ranks[column] = rank
            for capacity in (.1, .2, .3):
                k = max(1, int(np.ceil(len(part) * capacity)))
                seed_sets[(column, capacity)] = set(part.plot_key.iloc[order[:k]])
        for capacity in (.1, .2, .3):
            overlaps = []
            for i, first in enumerate(seeds):
                for second in seeds[i + 1:]:
                    a, b = seed_sets[(first, capacity)], seed_sets[(second, capacity)]
                    overlaps.append(len(a & b) / max(1, len(a | b)))
            stability.append({'site': site, 'input_set': input_set, 'capacity': capacity,
                              'mean_pairwise_jaccard': float(np.mean(overlaps))})
            k = max(1, int(np.ceil(len(part) * capacity)))
            table = part.copy()
            table['held_out_site'] = 'randomized trial-block fold'
            table['input_set'] = input_set
            table['capacity'] = capacity
            table['predicted_yield'] = table.pred_seed_42
            table['rank'] = seed_ranks['pred_seed_42']
            table['selected'] = table['rank'] <= k
            table['visit_frequency'] = np.mean(np.vstack([
                seed_ranks[column] <= k for column in seeds]), axis=0)
            table['rank_sd_across_seeds'] = np.std(np.vstack(
                [seed_ranks[column] for column in seeds]), axis=0)
            table['interval_low_80'] = np.nan
            table['interval_high_80'] = np.nan
            app_rows.append(table)

    # Keep the seed-42 score as the main result. Keep all seeds for rank stability.
    metric_frame = pd.DataFrame(metrics)
    predictions.to_csv(output / 'oof_predictions_by_seed.csv', index=False)
    pd.concat(app_rows, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    metric_frame.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(selection_records).to_csv(output / 'inner_selection_scores.csv', index=False)
    predictions[['fold', 'input_set', 'estimator']].drop_duplicates().to_csv(
        output / 'selected_models.csv', index=False)
    pd.DataFrame(stability).to_csv(output / 'rank_stability.csv', index=False)
    aggregate = metric_frame.groupby(['model', 'capacity']).apply(
        lambda part: pd.Series({
            'equal_site_mae': part.groupby('site').mae.mean().mean(),
            'plot_weighted_mae': np.average(part.mae, weights=part.n),
            'equal_site_low_yield_recall': part.groupby('site').low_yield_recall.mean().mean(),
            'equal_site_random_recall': part.groupby('site').random_recall.mean().mean(),
        }), include_groups=False).reset_index()
    aggregate.to_csv(output / 'aggregate_metrics.csv', index=False)
    try:
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = 'unknown'
    metadata = {
        'schema_version': 1, 'season': 2022, 'cutoff': cutoff.date().isoformat(),
        'split_strategy': 'nested randomized GroupKFold by location, experiment, and block',
        'selection_objective': objective,
        'image_trend_features_enabled': include_image_trends,
        'image_trend_features': trend_cols,
        'catboost_mae_candidate_enabled': include_catboost_mae,
        'display_protocol': 'Randomized trial-block holdout within known sites',
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seed': seed,
        'model_seed_values': [42, 101, 2026],
        'candidate_estimators': [name for name, _ in _candidates(
            ['poundsOfNitrogenPerAcre'], ['location_key'], cores, seed,
            include_catboost_mae=include_catboost_mae)],
        'group_count': group_count, 'eligible_plots': len(data),
        'sites': sorted(data.location_key.unique()), 'cores_used': cores,
        'site_feature_in_all_models': True,
        'limitations': 'This estimates interpolation to unseen trial blocks at known sites in one season. It does not test future seasons or new sites.',
        'feature_hash': _hash(features),
        'plot_table_hash': _hash(root / 'data/processed/plot_records.csv'),
        'code_revision': revision,
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__,
        'elapsed_seconds': round(time.time() - started, 1),
    }
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'inner_selection_scores.partial.csv').unlink(missing_ok=True)
    (output / 'predictions.partial.csv').unlink(missing_ok=True)
    print(f'Saved randomized grouped evaluation to {output}', flush=True)
