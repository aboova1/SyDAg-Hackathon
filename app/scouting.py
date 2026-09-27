"""Historical replay for a crew with limited scouting time."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Field Scout | IoT4Ag', layout='wide')
ROOT = Path(__file__).resolve().parents[1]
st.title('Field Scout')
st.write('Choose which maize plots to inspect while the trial is still in the field.')

runs = [p for p in (ROOT / 'outputs').rglob('run.json')
        if json.loads(p.read_text()).get('demo_ready')]
primary_demo = ROOT / 'outputs/demo_rank_aug15/run.json'
runs.sort(key=lambda p: (p != primary_demo, str(p)))
if not runs:
    st.info('No saved model results are available yet.')
    st.code('See README.md for the three commands that build the historical demo.')
    st.stop()
def run_label(path):
    metadata = json.loads(path.read_text())
    if path == primary_demo:
        return 'MAIN DEMO · Aug 15, 2022 · direct rank'
    return f"Research run · {metadata['cutoff']} · {path.parent.name}"

run_path = st.sidebar.selectbox('Historical decision run', runs, format_func=run_label)
metadata = json.loads(run_path.read_text())
priority_method = metadata.get('priority_method')
all_predictions = pd.read_csv(run_path.parent / 'predictions.csv', parse_dates=['collection_date'])
metrics = pd.read_csv(run_path.parent / 'metrics.csv')
sites = sorted(all_predictions.location_key.unique())
default_site = sites.index('Ames') if 'Ames' in sites else 0
site = st.sidebar.selectbox('Trial site', sites, index=default_site)
site_rows = all_predictions[all_predictions.location_key == site]
input_sets = sorted(site_rows.input_set.unique())
preferred = ('fields_indices_plus_site' if 'fields_indices_plus_site' in input_sets else
             ('combined_indices' if 'combined_indices' in input_sets else
             ('images' if 'images' in input_sets else
              ('combined' if 'combined' in input_sets else input_sets[0]))))
input_set = st.sidebar.selectbox('Information used', input_sets, index=input_sets.index(preferred))
set_rows = site_rows[site_rows.input_set == input_set]
capacities = sorted(set_rows.capacity.unique())
default_capacity = capacities.index(0.2) if 0.2 in capacities else 0
capacity = st.sidebar.selectbox('Crew capacity', capacities, index=default_capacity,
                                format_func=lambda value: f'{value:.0%} of this site')
max_image_age = st.sidebar.slider('Fresh image limit (days)', 7, 21, 14)
plots = set_rows[set_rows.capacity == capacity].sort_values('rank').copy()
k = int(plots.selected.sum())
priority = plots.head(k).copy()

st.caption(f"Historical replay · {metadata['cutoff']} · Yield in bushels per acre · "
           f"{metadata.get('display_protocol', 'Held-out site')}")
if priority_method:
    st.warning('The visit order uses a direct yield-rank score. The bu/acre estimate comes from a separate model. This score cannot diagnose crop stress.')
else:
    st.warning('This replay ranks low predicted yield for review. It cannot diagnose the cause of crop stress.')
stale_priority = priority.img_age_days > max_image_age
display_priority = priority.copy()
display_priority['status'] = np.where(stale_priority, 'Image is stale; inspect by eye', 'Image is recent enough')
visit_reason = ('Among the highest-priority yield ranks for this site and date'
                if priority_method else 'Among the lowest predictions for this site and date')
display_priority['visit_reason'] = np.where(stale_priority,
    'Image exceeds the freshness limit; check current field conditions',
    visit_reason)
display_priority.loc[stale_priority, 'predicted_yield'] = np.nan
col1, col2, col3 = st.columns(3)
col1.metric('Plots to visit', f'{k} / {len(plots)}')
col2.metric('Decision date', metadata['cutoff'])
fresh_priority = priority[~stale_priority]
col3.metric('Visit cutoff', f"{fresh_priority.predicted_yield.max():.1f} bu/acre"
           if len(fresh_priority) else 'Fresh estimate withheld')
st.caption('This cutoff changes with crew capacity. It ranks plots for a human check; it is not a crop-damage threshold.')

st.subheader('Visit list')
visit_columns = ['rank', 'plot_key', 'genotype', 'priority_score', 'predicted_yield', 'img_age_days',
                 'img_observation_count', 'visit_frequency', 'visit_reason']
shown_columns = [column for column in visit_columns + ['status'] if column in display_priority.columns]
st.dataframe(display_priority[shown_columns], hide_index=True, use_container_width=True)
st.caption('Visit frequency shows how often each plot ranked within capacity across three model seeds.')

csv = display_priority[shown_columns].to_csv(index=False)
st.download_button('Download visit list', csv, 'field-scout-visit-list.csv', 'text/csv')
order_note = ('Use the direct rank score for visit order. Read bu/acre as a separate estimate.'
              if priority_method else 'Rank plots by predicted yield.')
memo = (f"Field Scout visit list\nSite: {site}\nDecision date: {metadata['cutoff']}\n"
        f"Crew capacity: {capacity:.0%} ({k} plots)\n"
        f"{order_note} Predictions support review; they do not diagnose crop stress.\n\n"
        + display_priority[shown_columns].to_string(index=False))
st.download_button('Download field memo', memo, 'field-scout-memo.txt', 'text/plain')

st.subheader('Trial layout')
experiment = st.selectbox('Experiment', sorted(plots.experiment_key.astype(str).unique()))
layout = plots[plots.experiment_key.astype(str) == experiment].copy()
layout['priority'] = np.where(layout.selected, 'Visit', 'Other')
st.scatter_chart(layout, x='row', y='range', color='priority')
st.caption('Schematic position using trial row and range. These values are not GPS coordinates.')

st.subheader('Plot evidence')
plot_key = st.selectbox('Plot', priority.plot_key.tolist() or plots.plot_key.tolist())
selected = plots[plots.plot_key == plot_key].iloc[0]
left, right, score_column = st.columns(3)
if selected.img_age_days > max_image_age:
    left.metric('Predicted yield', 'Withheld: stale image')
    st.warning('Use this plot as a field check. Do not rely on its old yield estimate.')
else:
    left.metric('Predicted yield', f"{selected.predicted_yield:.1f} bu/acre")
right.metric('Prediction rank', f"{int(selected['rank'])} of {len(plots)}")
if priority_method and 'priority_score' in selected:
    score_column.metric('Priority score', f"{selected.priority_score:.3f}",
                        help='Lower values mean a higher scouting priority.')
if selected.img_age_days <= max_image_age and pd.notna(selected.interval_low_80):
    st.write(f"Residual band: {selected.interval_low_80:.1f} to {selected.interval_high_80:.1f} bu/acre")
    st.caption('This band is a research estimate. It has no finite-sample coverage guarantee.')

@st.cache_data(show_spinner=False)
def image_history(plot_key, season, cutoff):
    import rasterio
    index = pd.read_csv(ROOT / 'data/processed/image_manifest.csv')
    dates = pd.read_csv(ROOT / 'data/processed/collection_dates.csv')
    index = index[(index.season == season) & (index.modality == 'satellite') &
                  (index.plot_key == plot_key) & (index.join_status == 'matched')]
    dates = dates[(dates.season == season) & (dates.sensor.str.lower() == 'satellite')]
    index = index.merge(dates[['season', 'location_key', 'time_point', 'collection_date']],
                        on=['season', 'location_key', 'time_point'], validate='many_to_one')
    index = index[index.collection_date <= cutoff].sort_values('collection_date')
    arrays = []
    for row in index.itertuples(index=False):
        with rasterio.open(ROOT / row.relative_path) as src:
            array = src.read([4, 1, 2]).astype('float32')
            masks = np.all(src.read_masks() > 0, axis=0)
        arrays.append((row.collection_date, row.time_point, array, masks))
    limits = []
    for channel in range(3):
        values = [array[channel][mask & (array[channel] > 0)]
                  for _, _, array, mask in arrays]
        values = [value for value in values if value.size]
        if values:
            limits.append(tuple(np.quantile(np.concatenate(values), [.02, .98])))
        else:
            limits.append((0, 0))
    result = []
    for date, tp, array, masks in arrays:
        rgb = np.zeros((array.shape[1], array.shape[2], 3), dtype=np.uint8)
        for channel, (low, high) in enumerate(limits):
            if high > low:
                rgb[:, :, channel] = np.clip(
                    (array[channel] - low) / (high - low) * 255, 0, 255).astype('uint8')
        result.append((date, tp, rgb))
    return result

try:
    history = image_history(plot_key, int(metadata['season']), metadata['cutoff'])
    if history:
        cols = st.columns(min(3, len(history)))
        for index, (date, tp, image) in enumerate(history):
            cols[index % len(cols)].image(image, caption=f'{date} · {tp} · NIR/red/green')
        st.caption('Images use one display scale per plot history. Color is for comparison, not diagnosis.')
    else:
        st.info('No matched satellite images were found for this plot before the decision date.')
except Exception as exc:
    st.info(f'Image preview is unavailable: {exc}')

st.checkbox('Reveal harvest results and model evidence', key='reveal')
if st.session_state.reveal:
    st.subheader('Harvest results')
    evidence_columns = ['plot_key', 'priority_score', 'predicted_yield', 'actual_yield']
    evidence_columns = [column for column in evidence_columns if column in plots.columns]
    st.dataframe(plots[evidence_columns], hide_index=True)
    st.subheader('Held-out site results')
    site_metrics = metrics[(metrics.site == site) & (metrics.capacity == capacity)]
    st.dataframe(site_metrics, hide_index=True)
    aggregate = pd.read_csv(run_path.parent / 'aggregate_metrics.csv')
    st.subheader('Average across held-out sites')
    st.dataframe(aggregate[aggregate.capacity == capacity], hide_index=True)
    st.json(metadata)
