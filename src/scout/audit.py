"""Read source indexes without third-party packages or changes to raw data."""
import csv
import json
from collections import Counter
from pathlib import Path

MISSING = {'', 'NA', 'NaN', 'nan'}


def read_table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit(root, output):
    root, output = Path(root), Path(output)
    plots = read_table(root / 'data/processed/plot_records.csv')
    images = read_table(root / 'data/processed/image_manifest.csv')
    dates = read_table(root / 'data/processed/collection_dates.csv')
    plot_keys = Counter(p['plot_key'] for p in plots if p['plot_key'] not in MISSING)
    date_keys = Counter((d['season'], d['location_key'], d['time_point'])
                        for d in dates if d['sensor'].lower() == 'satellite')
    satellite = [i for i in images if i['modality'] == 'satellite']
    image_keys = Counter((i['plot_key'], i['time_point']) for i in satellite)
    report = {
        'plot_rows': len(plots),
        'missing_plot_key_rows': [{'season': p['season'], 'source_row_number': p['source_row_number']}
                                  for p in plots if p['plot_key'] in MISSING],
        'duplicate_plot_keys': [k for k, n in plot_keys.items() if n > 1],
        'duplicate_satellite_keys': [list(k) for k, n in image_keys.items() if n > 1],
        'duplicate_date_keys': [list(k) for k, n in date_keys.items() if n > 1],
        'missing_image_files': [i['relative_path'] for i in images
                                if not (root / i['relative_path']).is_file()],
        'satellite_images_without_dates': sum(
            (i['season'], i['location_key'], i['time_point']) not in date_keys for i in satellite),
        'join_status': dict(Counter(i['join_status'] for i in images)),
        'seasons': {},
    }
    matched = {i['plot_key'] for i in satellite if i['join_status'] == 'matched'}
    for season in sorted({p['season'] for p in plots}):
        rows = [p for p in plots if p['season'] == season]
        report['seasons'][season] = {
            'rows': len(rows), 'sites': dict(Counter(p['location_key'] for p in rows)),
            'known_yields': sum(p['yieldPerAcre'] not in MISSING for p in rows),
            'matched_satellite_rows': sum(p['plot_key'] in matched for p in rows),
            'eligible_labeled_rows': sum(p['plot_key'] in matched and p['yieldPerAcre'] not in MISSING for p in rows),
            'missing_by_field': {k: sum(p[k] in MISSING for p in rows) for k in rows[0]},
        }
    report['exclusions_by_site'] = {}
    for season in sorted({p['season'] for p in plots}):
        for site in sorted({p['location_key'] for p in plots if p['season'] == season}):
            subset = [p for p in plots if p['season'] == season and p['location_key'] == site]
            report['exclusions_by_site'][f'{season}|{site}'] = {
                'rows': len(subset),
                'missing_yield': sum(p['yieldPerAcre'] in MISSING for p in subset),
                'missing_plot_key': sum(p['plot_key'] in MISSING for p in subset),
                'labeled_without_image': sum(p['yieldPerAcre'] not in MISSING and
                                             p['plot_key'] not in matched for p in subset),
            }
    report['structural_pass'] = not any(report[k] for k in [
        'duplicate_plot_keys', 'duplicate_satellite_keys', 'duplicate_date_keys',
        'missing_image_files', 'satellite_images_without_dates'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report
