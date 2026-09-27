"""Yield error and scouting-capacity measures."""
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score_predictions(y, predicted, keys, site, model, interval_radius=None):
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keys = np.asarray(keys, dtype=str)
    order = np.lexsort((keys, predicted))
    low = y <= np.quantile(y, .2)
    rows = []
    for capacity in (.1, .2, .3):
        k = max(1, math.ceil(len(y) * capacity))
        selected = order[:k]
        random_recall = k / len(y)
        n_low = int(low.sum())
        recall = float(low[selected].sum() / n_low) if n_low else float('nan')
        row = {'site': site, 'model': model, 'n': len(y), 'capacity': capacity,
               'scout_count': k, 'low_yield_count': n_low,
               'mae': float(mean_absolute_error(y, predicted)),
               'rmse': float(np.sqrt(mean_squared_error(y, predicted))),
               'low_yield_recall': recall, 'low_yield_precision': float(low[selected].mean()),
               'random_recall': random_recall,
               'recall_lift': recall / random_recall if random_recall else float('nan')}
        if interval_radius is not None:
            row['interval_coverage'] = float(np.mean(np.abs(y - predicted) <= interval_radius))
            row['interval_width'] = float(2 * interval_radius)
        rows.append(row)
    return rows, order
