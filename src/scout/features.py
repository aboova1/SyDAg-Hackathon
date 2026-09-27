"""Extract per-image summaries without reading yield labels."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time

BANDS = ['red', 'green', 'blue', 'nir', 'rededge', 'deepblue']
BAND_TOKENS = ['red', 'green', 'blue', 'nir', 'red edge', 'deep blue']


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=output.name, suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        frame.to_csv(temp_name, index=False)
        os.replace(temp_name, output)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def extract(root, output, season=2022, force=False):
    import numpy as np
    import pandas as pd
    import rasterio

    root, output = Path(root), Path(output)
    index = pd.read_csv(root / 'data/processed/image_manifest.csv')
    index = index[(index.season == season) & (index.modality == 'satellite') &
                  (index.join_status == 'matched')].copy()
    dates = pd.read_csv(root / 'data/processed/collection_dates.csv')
    dates = dates[dates.sensor.str.lower() == 'satellite']
    index = index.merge(dates[['season', 'location_key', 'time_point', 'collection_date']],
                        on=['season', 'location_key', 'time_point'], validate='many_to_one')
    if index.empty or index.collection_date.isna().any():
        raise ValueError('No images or missing collection dates. Run the audit first.')

    source_hashes = {row.relative_path: _digest(root / row.relative_path)
                     for row in index.itertuples(index=False)}
    settings = {'feature_schema_version': 2, 'season': int(season), 'band_order': BANDS, 'band_validation': BAND_TOKENS,
                'mask_rule': 'all six GDAL masks valid; exclude all-zero pixels',
                'indices': {'ndvi': [3, 0], 'ndre': [3, 4], 'gndvi': [3, 1]},
                'source_sha256': source_hashes}
    audit_path = output.with_suffix('.audit.json')
    if output.exists() and audit_path.exists() and not force:
        old = json.loads(audit_path.read_text())
        if old.get('settings') == settings and not old.get('failed'):
            print(f'Using verified cache: {output}', flush=True)
            return
        raise ValueError('Feature cache does not match current source data or settings. Use --force.')

    started = time.time()
    rows, errors, descriptions = [], [], set()
    for number, item in enumerate(index.itertuples(index=False), 1):
        try:
            with rasterio.open(root / item.relative_path) as src:
                if src.count != 6:
                    raise ValueError(f'Expected six bands; found {src.count}')
                desc = [str(value or '').lower() for value in src.descriptions]
                if len(desc) != 6 or any(token not in value for token, value in zip(BAND_TOKENS, desc)):
                    raise ValueError(f'Band names/order conflict: {src.descriptions}')
                descriptions.add(tuple(desc))
                if src.nodata not in (0, 0.0):
                    raise ValueError(f'Unexpected no-data value: {src.nodata}')
                pixels = src.read().astype('float64')
                valid = np.all(src.read_masks() > 0, axis=0) & np.all(np.isfinite(pixels), axis=0)
                valid &= np.any(pixels != 0, axis=0)
            if valid.sum() < 5:
                raise ValueError(f'Only {int(valid.sum())} valid pixels')
            row = {'plot_key': item.plot_key, 'location_key': item.location_key,
                   'collection_date': item.collection_date, 'time_point': item.time_point,
                   'img_valid_fraction': float(valid.mean()),
                   'img_valid_pixels': int(valid.sum()),
                   'img_saturated_fraction': float(np.mean(pixels == 65535)),
                   'img_zero_fraction': float(np.mean(np.all(pixels == 0, axis=0))),
                   'source_path': item.relative_path,
                   'source_sha256': source_hashes[item.relative_path]}
            for band, values in zip(BANDS, pixels[:, valid]):
                for name, value in [('median', np.median(values)), ('std', np.std(values)),
                                    ('p10', np.quantile(values, .1)), ('p90', np.quantile(values, .9))]:
                    row[f'img_{band}_{name}'] = float(value)
            for name, a, b in [('ndvi', 3, 0), ('ndre', 3, 4), ('gndvi', 3, 1)]:
                denominator = pixels[a, valid] + pixels[b, valid]
                usable = denominator != 0
                values = (pixels[a, valid][usable] - pixels[b, valid][usable]) / denominator[usable]
                if len(values) == 0:
                    raise ValueError(f'No valid denominator for {name}')
                row[f'img_{name}_median'] = float(np.median(values))
                row[f'img_{name}_std'] = float(np.std(values))
            rows.append(row)
        except Exception as exc:
            errors.append({'path': item.relative_path, 'error': str(exc)})
        if number % 1000 == 0 or number == len(index):
            print(f'Images: {number}/{len(index)}; usable: {len(rows)}; failed: {len(errors)}', flush=True)

    if errors:
        _atomic_csv(pd.DataFrame(rows), output)
        report = {'settings': settings, 'written': len(rows), 'failed': errors,
                  'observed_descriptions': [list(row) for row in sorted(descriptions)],
                  'elapsed_seconds': round(time.time() - started, 1)}
        audit_path.write_text(json.dumps(report, indent=2) + '\n')
        raise RuntimeError(f'{len(errors)} images failed. Review {audit_path}.')
    feature_frame = pd.DataFrame(rows)
    _atomic_csv(feature_frame, output)
    if not feature_frame.empty:
        quality = feature_frame.groupby(['location_key', 'collection_date'], as_index=False).agg(
            images=('plot_key', 'count'), median_valid_fraction=('img_valid_fraction', 'median'),
            median_zero_fraction=('img_zero_fraction', 'median'),
            median_saturated_fraction=('img_saturated_fraction', 'median'),
            median_ndvi=('img_ndvi_median', 'median'))
        _atomic_csv(quality, output.with_name('image_quality_by_site_date.csv'))
    report = {'settings': settings, 'written': len(rows), 'failed': [],
              'observed_descriptions': [list(row) for row in sorted(descriptions)],
              'digital_number_note': 'Values are digital numbers, not verified surface reflectance.',
              'elapsed_seconds': round(time.time() - started, 1)}
    audit_path.write_text(json.dumps(report, indent=2) + '\n')
