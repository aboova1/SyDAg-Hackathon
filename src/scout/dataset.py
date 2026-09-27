"""Build date-safe model cohorts and record every excluded plot."""
from pathlib import Path
import pandas as pd

MISSING = {'', 'NA', 'NaN', 'nan'}


def build_dataset(root, features_path, season, cutoff):
    root, cutoff = Path(root), pd.Timestamp(cutoff)
    if cutoff.year != int(season):
        raise ValueError('Cutoff year must match the selected season.')
    plots = pd.read_csv(root / 'data/processed/plot_records.csv', dtype={'plot_key': str})
    plots = plots[plots.season == int(season)].copy()
    plots['yieldPerAcre'] = pd.to_numeric(plots.yieldPerAcre, errors='coerce')
    image_features = pd.read_csv(features_path, parse_dates=['collection_date'])
    image_features = image_features[image_features.collection_date <= cutoff].copy()
    if image_features.duplicated(['plot_key', 'collection_date']).any():
        raise ValueError('Duplicate plot-date feature rows.')
    latest = image_features.sort_values(['plot_key', 'collection_date']).groupby('plot_key').tail(1).copy()
    first = image_features.sort_values(['plot_key', 'collection_date']).groupby('plot_key').head(1).set_index('plot_key')
    for name in ['ndvi', 'ndre', 'gndvi']:
        col = f'img_{name}_median'
        latest[f'img_{name}_change'] = latest[col] - latest.plot_key.map(first[col])
    counts = image_features.groupby('plot_key').size()
    latest['img_observation_count'] = latest.plot_key.map(counts)
    latest['img_age_days'] = (cutoff - latest.collection_date).dt.days
    latest['img_days_since_first'] = latest.plot_key.map(
        image_features.groupby('plot_key').collection_date.min().map(lambda d: (cutoff - d).days))

    plot_sites = plots.set_index('plot_key').location_key
    mismatched_sites = latest.apply(lambda row: row.location_key != plot_sites.get(row.plot_key), axis=1)
    if mismatched_sites.any():
        raise ValueError(f'{int(mismatched_sites.sum())} image rows have a site mismatch.')
    latest = latest.drop(columns=['location_key'])
    latest_keys = set(latest.plot_key)
    exclusions = []
    eligible = []
    for row in plots.itertuples(index=False):
        if pd.isna(row.yieldPerAcre):
            reason = 'missing_yield'
        elif row.plot_key in MISSING:
            reason = 'missing_plot_key'
        elif row.plot_key not in latest_keys:
            reason = 'no_satellite_image_by_cutoff'
        else:
            eligible.append(row.plot_key)
            continue
        exclusions.append({'season': season, 'location_key': row.location_key,
                           'plot_key': row.plot_key, 'reason': reason})
    plots = plots[plots.plot_key.isin(eligible)].copy()
    data = plots.merge(latest, on='plot_key', validate='one_to_one')
    data['planting_day'] = pd.to_datetime(data.plantingDate, format='mixed', errors='coerce').dt.dayofyear
    for col in ['poundsOfNitrogenPerAcre', 'irrigationProvided']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['genotype'] = data.genotype.fillna('unknown').astype(str)
    image_cols = [col for col in latest if col.startswith('img_') and col not in {
        'img_source_path', 'img_source_sha256'}]
    if any(col in image_cols for col in ['img_source_path', 'img_source_sha256']):
        raise ValueError('Image provenance fields cannot be model inputs.')
    forbidden = {'yieldPerAcre', 'plot_key', 'row', 'range', 'plotNumber', 'block',
                 'experiment_key', 'daysToAnthesis', 'GDDToAnthesis', 'totalStandCount'}
    if forbidden.intersection(image_cols):
        raise ValueError('Target, plot identity, or undated field leaked into image features.')
    if data.empty:
        raise ValueError(f'No labeled plots have imagery by {cutoff.date()}.')
    return data, pd.DataFrame(exclusions), image_cols


def build_dataset_by_dap(root, features_path, season, days_after_planting):
    """Build a plot table from each plot's latest image by its DAP cutoff."""
    root = Path(root)
    days_after_planting = int(days_after_planting)
    if days_after_planting < 0:
        raise ValueError('Days after planting must be zero or greater.')

    plots = pd.read_csv(root / 'data/processed/plot_records.csv', dtype={'plot_key': str})
    plots = plots[plots.season == int(season)].copy()
    plots['yieldPerAcre'] = pd.to_numeric(plots.yieldPerAcre, errors='coerce')
    plots['plantingDate'] = pd.to_datetime(
        plots.plantingDate, format='mixed', errors='coerce')
    image_features = pd.read_csv(features_path, parse_dates=['collection_date'])
    if image_features.duplicated(['plot_key', 'collection_date']).any():
        raise ValueError('Duplicate plot-date feature rows.')
    image_features = image_features.merge(
        plots[plots.plot_key.notna()][['plot_key', 'location_key', 'plantingDate']],
        on='plot_key', how='left', validate='many_to_one', suffixes=('', '_plot'))
    image_features['img_days_after_planting'] = (
        image_features.collection_date - image_features.plantingDate).dt.days
    image_features = image_features[
        image_features.img_days_after_planting.between(0, days_after_planting)].copy()

    latest = image_features.sort_values(
        ['plot_key', 'collection_date']).groupby('plot_key').tail(1).copy()
    first = image_features.sort_values(
        ['plot_key', 'collection_date']).groupby('plot_key').head(1).set_index('plot_key')
    for name in ['ndvi', 'ndre', 'gndvi']:
        col = f'img_{name}_median'
        latest[f'img_{name}_change'] = latest[col] - latest.plot_key.map(first[col])
    counts = image_features.groupby('plot_key').size()
    latest['img_observation_count'] = latest.plot_key.map(counts)
    latest['img_age_days'] = days_after_planting - latest.img_days_after_planting
    latest['img_days_since_first'] = days_after_planting - latest.plot_key.map(
        image_features.groupby('plot_key').img_days_after_planting.min())

    site_lookup = plots.set_index('plot_key').location_key
    mismatched_sites = latest.apply(
        lambda row: row.location_key != site_lookup.get(row.plot_key), axis=1)
    if mismatched_sites.any():
        raise ValueError(f'{int(mismatched_sites.sum())} image rows have a site mismatch.')
    latest = latest.drop(columns=['location_key', 'location_key_plot', 'plantingDate'])
    latest_keys = set(latest.plot_key)
    exclusions, eligible = [], []
    for row in plots.itertuples(index=False):
        if pd.isna(row.yieldPerAcre):
            reason = 'missing_yield'
        elif row.plot_key in MISSING:
            reason = 'missing_plot_key'
        elif pd.isna(row.plantingDate):
            reason = 'missing_planting_date'
        elif row.plot_key not in latest_keys:
            reason = 'no_satellite_image_by_dap_cutoff'
        else:
            eligible.append(row.plot_key)
            continue
        exclusions.append({'season': season, 'location_key': row.location_key,
                           'plot_key': row.plot_key, 'reason': reason})

    plots = plots[plots.plot_key.isin(eligible)].copy()
    data = plots.merge(latest, on='plot_key', validate='one_to_one')
    data['planting_day'] = data.plantingDate.dt.dayofyear
    for col in ['poundsOfNitrogenPerAcre', 'irrigationProvided']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['genotype'] = data.genotype.fillna('unknown').astype(str)
    image_cols = [col for col in latest if col.startswith('img_') and col not in {
        'img_source_path', 'img_source_sha256'}]
    forbidden = {'yieldPerAcre', 'plot_key', 'row', 'range', 'plotNumber', 'block',
                 'experiment_key', 'daysToAnthesis', 'GDDToAnthesis', 'totalStandCount'}
    if forbidden.intersection(image_cols):
        raise ValueError('Target, plot identity, or undated field leaked into image features.')
    if data.empty:
        raise ValueError(f'No labeled plots have imagery by {days_after_planting} DAP.')
    return data, pd.DataFrame(exclusions), image_cols
