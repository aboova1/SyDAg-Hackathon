"""Nested site-held-out yield comparisons with date-safe inputs."""
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
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .audit import audit
from .dataset import build_dataset
from .metrics import score_predictions
from .splits import outer_sites, inner_sites


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _prepare(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(('category', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder='drop')


class _CatBoostFrame(BaseEstimator, TransformerMixin):
    """Keep named columns and prepare category strings for CatBoost."""

    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        columns = self.numeric + self.categorical
        frame = X.loc[:, columns].copy()
        for column in self.numeric:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
        for column in self.categorical:
            frame[column] = frame[column].fillna('unknown').astype(str)
        return frame


def _candidates(numeric, categorical, cores, seed=42, include_catboost_mae=False,
                include_catboost_native=False):
    features = _prepare(numeric, categorical)
    candidates = [
        ('ridge_a1', Pipeline([('inputs', features), ('model', Ridge(alpha=1.0))])),
        ('ridge_a100', Pipeline([('inputs', _prepare(numeric, categorical)), ('model', Ridge(alpha=100.0))])),
        ('extra_trees_leaf5', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                           max_features=.8, random_state=seed, n_jobs=cores))])),
        ('extra_trees_leaf15', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', ExtraTreesRegressor(n_estimators=400, min_samples_leaf=15,
                                           max_features=.8, random_state=seed, n_jobs=cores))])),
        ('catboost_depth6', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', CatBoostRegressor(iterations=500, depth=6, learning_rate=.05,
                                        loss_function='RMSE', l2_leaf_reg=5, random_seed=seed,
                                        thread_count=cores, verbose=False, allow_writing_files=False))])),
        ('catboost_depth4', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', CatBoostRegressor(iterations=500, depth=4, learning_rate=.05,
                                        loss_function='RMSE', l2_leaf_reg=5, random_seed=seed,
                                        thread_count=cores, verbose=False, allow_writing_files=False))])),
        ('catboost_depth8', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', CatBoostRegressor(iterations=500, depth=8, learning_rate=.05,
                                        loss_function='RMSE', l2_leaf_reg=5, random_seed=seed,
                                        thread_count=cores, verbose=False, allow_writing_files=False))])),
        ('hist_gradient_leaf15', Pipeline([('inputs', _prepare(numeric, categorical)),
             ('model', HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
                    max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                    early_stopping=False, random_state=seed))])),
    ]
    if include_catboost_mae:
        candidates.append(('catboost_mae_depth6', Pipeline([
            ('inputs', _prepare(numeric, categorical)),
            ('model', CatBoostRegressor(iterations=500, depth=6, learning_rate=.05,
                                       loss_function='MAE', l2_leaf_reg=5, random_seed=seed,
                                       thread_count=cores, verbose=False,
                                       allow_writing_files=False)),
        ])))
    if include_catboost_native:
        for depth in (6, 8):
            candidates.append((f'catboost_native_depth{depth}', Pipeline([
                ('inputs', _CatBoostFrame(numeric, categorical)),
                ('model', CatBoostRegressor(
                    iterations=500, depth=depth, learning_rate=.05,
                    loss_function='RMSE', l2_leaf_reg=5,
                    random_seed=seed, thread_count=cores, verbose=False,
                    allow_writing_files=False, cat_features=categorical)),
            ])))
    return candidates


def _fold_predictions(site, train, validation, numeric, categorical, cores, seed=42):
    results = {}
    for name, estimator in _candidates(numeric, categorical, cores, seed):
        estimator.fit(train, train.yieldPerAcre)
        pred = estimator.predict(validation)
        mae = float(np.mean(np.abs(validation.yieldPerAcre.to_numpy() - pred)))
        results[name] = {'mae': mae, 'prediction': pred}
    return results


def evaluate(root, features, cutoff, output, season=2022, capacities=(.1, .2, .3), seed=42, required_plot_keys=None):
    root, features, output = Path(root), Path(features), Path(output)
    cutoff_date = pd.Timestamp(cutoff)
    if cutoff_date.year != season:
        raise ValueError('Cutoff year must match the selected season.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder for every model run.')
    if isinstance(capacities, (int, float)):
        capacities = (float(capacities),)
    else:
        capacities = tuple(capacities)
    for capacity in capacities:
        if not 0 < capacity <= 1:
            raise ValueError('Capacity must be above zero and at most one.')
    source_audit = audit(root, output / 'source_audit.json')
    if not source_audit['structural_pass']:
        raise ValueError('The source audit failed. Review source_audit.json.')
    extraction = json.loads(features.with_suffix('.audit.json').read_text())
    if extraction['failed'] or extraction['settings']['season'] != season:
        raise ValueError('Feature audit failed or season does not match.')

    started = time.time()
    data, exclusions, image_cols = build_dataset(root, features, season, cutoff_date)
    if data.location_key.nunique() < 3:
        raise ValueError('Need at least three sites for grouped evaluation.')
    if required_plot_keys is not None:
        missing = set(required_plot_keys) - set(data.plot_key)
        if missing:
            raise ValueError(f'{len(missing)} common plots are not eligible at {cutoff_date.date()}.')
        data = data[data.plot_key.isin(required_plot_keys)].copy()
    output.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(output / 'exclusions.csv', index=False)
    data[['plot_key', 'location_key', 'collection_date']].to_csv(output / 'eligible_plots.csv', index=False)
    field_numeric = ['poundsOfNitrogenPerAcre', 'irrigationProvided', 'planting_day']
    indices = [c for c in image_cols if c.startswith(('img_ndvi_', 'img_ndre_', 'img_gndvi_'))
               or c in {'img_valid_fraction', 'img_observation_count', 'img_age_days', 'img_days_since_first'}]
    groups = {
        'nitrogen': (['poundsOfNitrogenPerAcre'], []),
        'fields': (field_numeric, ['genotype']),
        'images': (image_cols, []),
        'indices': (indices, []),
        'combined': (field_numeric + image_cols, ['genotype']),
        'combined_indices': (field_numeric + indices, ['genotype']),
    }
    cores = os.cpu_count() or 1
    all_metrics, all_predictions, model_choices, rank_stabilities = [], [], [], []
    outer_count = len(data.location_key.unique())
    for outer_index, (site, train, hold) in enumerate(outer_sites(data), 1):
        print(f'Outer site {outer_index}/{outer_count}: {site}', flush=True)
        y_train = train.yieldPerAcre.to_numpy()
        baseline = np.full(len(hold), float(np.mean(y_train)))
        base_metrics, base_order = score_predictions(hold.yieldPerAcre, baseline,
                                                      hold.plot_key, site, 'mean_reference')
        all_metrics.extend(row for row in base_metrics if row['capacity'] in capacities)
        base_rank = np.empty(len(base_order), dtype=int)
        base_rank[base_order] = np.arange(1, len(base_order) + 1)
        for capacity in capacities:
            base_table = hold[['plot_key', 'location_key', 'experiment_key', 'row', 'range',
                               'genotype', 'collection_date', 'img_age_days',
                               'img_observation_count']].copy()
            base_table['held_out_site'] = site
            base_table['input_set'] = 'mean_reference'
            base_table['estimator'] = 'training_mean'
            base_table['capacity'] = capacity
            base_table['predicted_yield'] = baseline
            base_table['actual_yield'] = hold.yieldPerAcre.to_numpy()
            base_table['interval_low_80'] = np.nan
            base_table['interval_high_80'] = np.nan
            base_table['rank'] = base_rank
            base_table['selected'] = base_rank <= max(1, int(np.ceil(len(hold) * capacity)))
            all_predictions.append(base_table)
        for variant, (numeric, categorical) in groups.items():
            candidate_errors = {name: [] for name, _ in _candidates(numeric, categorical, cores, seed)}
            candidate_residuals = {name: [] for name in candidate_errors}
            for inner_site, inner_train, inner_valid in inner_sites(train, site):
                fold = _fold_predictions(inner_site, inner_train, inner_valid,
                                         numeric, categorical, cores, seed)
                for name, record in fold.items():
                    candidate_errors[name].append(record['mae'])
                    candidate_residuals[name].append(np.abs(
                        inner_valid.yieldPerAcre.to_numpy() - record['prediction']))
            mean_errors = {name: float(np.mean(scores)) for name, scores in candidate_errors.items()}
            selected_name = min(mean_errors, key=lambda name: (mean_errors[name], name))
            grouped_residuals = candidate_residuals[selected_name]
            group_scores = sorted(float(np.max(residuals)) for residuals in grouped_residuals)
            conformal_rank = min(len(group_scores), int(np.ceil((len(group_scores) + 1) * .8)))
            radius = float(group_scores[conformal_rank - 1])
            seed_values = sorted(set([seed, 101, 2026]))
            seed_predictions = {}
            for model_seed in seed_values:
                estimator = dict(_candidates(numeric, categorical, cores, model_seed))[selected_name]
                estimator.fit(train, y_train)
                seed_predictions[model_seed] = estimator.predict(hold)
            predicted = seed_predictions[seed]
            seed_orders = {model_seed: np.lexsort((hold.plot_key.astype(str).to_numpy(), values))
                           for model_seed, values in seed_predictions.items()}
            names = [variant]
            predictions = [predicted]
            intervals = [radius]
            model_choices.append({'held_out_site': site, 'input_set': variant,
                                  'selected_estimator': selected_name,
                                  'inner_site_mean_mae': mean_errors[selected_name],
                                  'inner_site_scores': mean_errors,
                                  'oof_site_count': len(grouped_residuals),
                                  'oof_group_max_scores': group_scores,
                                  'interval_radius_80': radius,
                                  'interval_method': '80% site-block split-conformal reference'})
            for model_name, yhat, interval in zip(names, predictions, intervals):
                metrics, order = score_predictions(hold.yieldPerAcre, yhat, hold.plot_key,
                                                   site, model_name, interval)
                for metric in metrics:
                    if metric['capacity'] in capacities:
                        all_metrics.append(metric)
                for capacity in capacities:
                    k = max(1, int(np.ceil(len(hold) * capacity)))
                    rank = np.empty(len(order), dtype=int)
                    rank[order] = np.arange(1, len(order) + 1)
                    selected_sets = {model_seed: set(hold.plot_key.iloc[seed_orders[model_seed][:k]])
                                     for model_seed in seed_values}
                    stability_scores = []
                    seed_keys = list(selected_sets)
                    for first_index in range(len(seed_keys)):
                        for second_index in range(first_index + 1, len(seed_keys)):
                            first_set, second_set = selected_sets[seed_keys[first_index]], selected_sets[seed_keys[second_index]]
                            stability_scores.append(len(first_set & second_set) / max(1, len(first_set | second_set)))
                    stable_keys = {key: sum(key in selected for selected in selected_sets.values()) / len(seed_values)
                                   for key in hold.plot_key.astype(str)}
                    seed_rank_std = np.std(np.vstack([
                        np.argsort(seed_orders[model_seed]) + 1 for model_seed in seed_values]), axis=0)
                    rank_stabilities.append({'site': site, 'model': model_name, 'capacity': capacity,
                                             'mean_pairwise_jaccard': float(np.mean(stability_scores)),
                                             'min_pairwise_jaccard': float(np.min(stability_scores))})
                    table = hold[['plot_key', 'location_key', 'experiment_key', 'row', 'range',
                                  'genotype', 'collection_date', 'img_age_days',
                                  'img_observation_count']].copy()
                    table['held_out_site'] = site
                    table['input_set'] = variant if model_name != 'mean_reference' else 'mean_reference'
                    table['estimator'] = selected_name if model_name != 'mean_reference' else 'training_mean'
                    table['capacity'] = capacity
                    table['predicted_yield'] = yhat
                    table['actual_yield'] = hold.yieldPerAcre.to_numpy()
                    table['interval_low_80'] = yhat - interval if interval is not None else np.nan
                    table['interval_high_80'] = yhat + interval if interval is not None else np.nan
                    table['rank'] = rank
                    table['selected'] = rank <= k
                    table['visit_frequency'] = table.plot_key.astype(str).map(stable_keys)
                    table['rank_sd_across_seeds'] = seed_rank_std
                    all_predictions.append(table)
        if outer_index % 1 == 0:
            print(f'Completed {outer_index}/{outer_count} outer sites', flush=True)

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / 'metrics.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(model_choices).to_json(output / 'model_selection.json', orient='records', indent=2)
    stability = pd.DataFrame(rank_stabilities)
    stability.to_csv(output / 'rank_stability.csv', index=False)
    aggregates = []
    for (model, capacity), group in metrics.groupby(['model', 'capacity']):
        site_equal = group.groupby('site').mae.mean().mean()
        row = {'model': model, 'capacity': capacity,
               'equal_site_mae': float(site_equal), 'equal_site_rmse': float(group.groupby('site').rmse.mean().mean()),
               'plot_weighted_mae': float(np.average(group.mae, weights=group.n)),
               'plot_weighted_rmse': float(np.sqrt(np.average(group.rmse ** 2, weights=group.n))),
               'equal_site_low_yield_recall': float(group.groupby('site').low_yield_recall.mean().mean()),
               'equal_site_random_recall': float(group.groupby('site').random_recall.mean().mean())}
        if 'interval_coverage' in group:
            row['equal_site_interval_coverage'] = float(group.groupby('site').interval_coverage.mean().mean())
            row['equal_site_interval_width'] = float(group.groupby('site').interval_width.mean().mean())
        stable = stability[(stability.model == model) & (stability.capacity == capacity)]
        if not stable.empty:
            row['mean_pairwise_visit_jaccard'] = float(stable.mean_pairwise_jaccard.mean())
            row['min_pairwise_visit_jaccard'] = float(stable.min_pairwise_jaccard.min())
        aggregates.append(row)
    pd.DataFrame(aggregates).to_csv(output / 'aggregate_metrics.csv', index=False)
    try:
        git_revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                               text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_revision = 'unknown'
    metadata = {'schema_version': 2, 'season': season, 'cutoff': cutoff_date.date().isoformat(),
                'capacities': list(capacities), 'seed': seed, 'cores_used': cores,
                'protocol': 'nested leave-one-site-out; inner site-fold model choice; 2022 development evidence',
                'outer_sites': sorted(data.location_key.unique()), 'plot_rows': source_audit['seasons'][str(season)]['rows'],
                'eligible_rows': len(data), 'eligible_sites': data.location_key.nunique(),
                'excluded_rows': len(exclusions), 'feature_count': len(image_cols),
                'features_sha256': _sha256(features),
                'plots_sha256': _sha256(root / 'data/processed/plot_records.csv'),
                'code_revision': git_revision, 'python': platform.python_version(),
                'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                'elapsed_seconds': round(time.time() - started, 1),
                'interval_note': '80% site-block split-conformal reference uses maxima from four inner held-out sites. Report outer coverage and width; five sites do not establish broad exchangeability.',
                'rank_seed_values': sorted(set([seed, 101, 2026])),
                'scouting_note': 'Low site-specific yield is a retrospective proxy, not a diagnosis or causal finding.'}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
