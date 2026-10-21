"""Checked local churn scoring and explicit review policies. No remote inference."""
import csv
import hashlib
import io
import json
import math
import re
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ['Call Failure', 'Complains', 'Subscription Length', 'Charge Amount',
            'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
            'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Age']
MODEL_NAMES = ['dummy', 'logistic', 'boosting', 'boosting_sigmoid']
EXCLUDED = ['Status', 'Customer Value', 'Churn']
LIMITS = {f: (0, 1_000_000_000) for f in FEATURES}
LIMITS.update({'Complains': (0, 1), 'Charge Amount': (0, 10),
               'Age Group': (1, 5), 'Tariff Plan': (1, 2), 'Age': (0, 120)})


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_data(path):
    frame = pd.read_csv(path)
    frame.columns = [' '.join(x.split()) for x in frame.columns]
    required = set(FEATURES + ['Churn', 'Status', 'Customer Value'])
    if set(frame.columns) != required or len(frame) == 0:
        raise ValueError('Unexpected UCI CSV schema or empty dataset')
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame['Churn'].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not set(y).issubset({0, 1}):
        raise ValueError('Non-finite inputs or non-binary labels')
    if not np.equal(x, np.floor(x)).all():
        raise ValueError('Expected integer UCI predictors')
    groups = np.array([sha256(json.dumps(row.tolist(), separators=(',', ':')).encode()) for row in x])
    return frame, x, y.astype(int), groups


def split_indices(x, y, groups, seed):
    all_i = np.arange(len(y))
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(x, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(x[dev], groups=groups[dev]))
    parts = {'fit': dev[a], 'calibration': dev[b], 'evaluation': test}
    if set(np.concatenate(list(parts.values()))) != set(all_i):
        raise AssertionError('Partitions are not exhaustive')
    for name, indices in parts.items():
        if len(set(y[indices])) != 2:
            raise ValueError(f'Both classes required in {name}; no split reroll')
    names = list(parts)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(groups[parts[name]]) & set(groups[parts[other]]):
                raise AssertionError('Duplicate predictor group crossed partitions')
    return parts


def checked_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite one-dimensional probabilities in [0,1]')
    return p


def exact_capacity(p, ids, fraction):
    p = checked_probabilities(p)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Capacity fraction must be within [0,1]')
    if len(ids) != len(p) or len(set(ids)) != len(ids):
        raise ValueError('Record IDs must be unique and aligned')
    k = int((Decimal(str(fraction)) * len(p)).to_integral_value(rounding=ROUND_CEILING))
    order = sorted(range(len(p)), key=lambda i: (-float(p[i]), ids[i]))
    flag = np.zeros(len(p), dtype=bool)
    flag[order[:k]] = True
    return flag


def development_cutoff(p, fraction=.1):
    p = checked_probabilities(p)
    if len(p) == 0 or not 0 < fraction <= 1:
        raise ValueError('Nonempty calibration scores and positive capacity required')
    k = int((Decimal(str(fraction)) * len(p)).to_integral_value(rounding=ROUND_CEILING))
    return float(np.sort(p)[::-1][k - 1])


def confusion(y, selected):
    y = np.asarray(y)
    selected = np.asarray(selected, dtype=bool)
    if y.shape != selected.shape or y.ndim != 1 or not set(y).issubset({0, 1}):
        raise ValueError('Aligned binary outcomes required')
    tp = int(np.sum((y == 1) & selected)); fp = int(np.sum((y == 0) & selected))
    fn = int(np.sum((y == 1) & ~selected)); tn = int(np.sum((y == 0) & ~selected))
    return {'n': len(y), 'selected': tp + fp, 'selected_fraction': (tp + fp) / len(y) if len(y) else 0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None}


def oracle_ap(y, p):
    """Independent threshold-group AP; ties contribute at their completed group precision."""
    buckets = {}
    for target, score in zip(y, p):
        bucket = buckets.setdefault(float(score), [0, 0])
        bucket[0] += int(target); bucket[1] += 1
    positives = sum(int(v) for v in y)
    if not positives:
        return 0.
    tp = seen = 0
    value = 0.
    for score in sorted(buckets, reverse=True):
        pos, count = buckets[score]; tp += pos; seen += count
        value += (pos / positives) * (tp / seen)
    return value


def reliability(y, p):
    """Ten fixed-width bins, with the final bin including probability one."""
    p = checked_probabilities(p)
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for i in range(10):
        use = bins == i
        if np.any(use):
            rows.append({'lower': i / 10, 'upper': (i + 1) / 10, 'n': int(use.sum()),
                         'mean_score': float(np.mean(p[use])), 'event_fraction': float(np.mean(y[use]))})
    return rows


def evaluate(y, p, ids, cutoff):
    p = checked_probabilities(p)
    ap = float(average_precision_score(y, p)); brier = float(brier_score_loss(y, p))
    independent_brier = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(y, p)) / len(y)
    if not math.isclose(ap, oracle_ap(y, p), abs_tol=1e-12) or not math.isclose(brier, independent_brier, abs_tol=1e-12):
        raise AssertionError('Independent metric disagreement')
    policies = {'cutoff_0.5': p >= .5, 'development_fixed_cutoff': p >= cutoff,
                'exact_10_percent': exact_capacity(p, ids, .1)}
    return {'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
            'ap': ap, 'brier': brier, 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'development_cutoff': cutoff, 'reliability_bins': reliability(y, p),
            'policies': {k: confusion(y, v) for k, v in policies.items()}}


def parse_csv(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 1_000_000:
        raise ValueError('CSV limit is 1 MB')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    allowed = set(FEATURES + ['record_id', 'Churn'])
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('CSV requires unique column names')
    if not set(FEATURES).issubset(reader.fieldnames) or set(reader.fieldnames) - allowed:
        raise ValueError('Use the eleven exact feature columns, optional record_id and optional Churn; Status/Customer Value are not accepted')
    rows = []
    ids = set()
    for i, row in enumerate(reader, 1):
        if i > 5000 or None in row or any(v is None for v in row.values()):
            raise ValueError('Maximum 5,000 rows; all rows must match the header')
        key = row.get('record_id', f'row-{i:06d}')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', key) or key in ids:
            raise ValueError('record_id must be unique, 1–64 letters, numbers, underscore or hyphen')
        ids.add(key)
        out = {'record_id': key}
        for f in FEATURES:
            try:
                val = float(row[f])
            except (ValueError, TypeError):
                raise ValueError(f'Row {i}: invalid {f}') from None
            low, high = LIMITS[f]
            if not math.isfinite(val) or not val.is_integer() or not low <= val <= high:
                raise ValueError(f'Row {i}: {f} must be an integer within [{low},{high}]')
            out[f] = int(val)
        if 'Churn' in row:
            if row['Churn'] not in ['0', '1']:
                raise ValueError('Optional Churn values must be 0 or 1')
            out['Churn'] = int(row['Churn'])
        rows.append(out)
    if not rows:
        raise ValueError('CSV has no data rows')
    return rows


def score_csv(text, bundle, model='boosting_sigmoid', capacity=.1):
    rows = parse_csv(text)
    if model not in MODEL_NAMES:
        raise ValueError('Unknown model')
    x = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    p = checked_probabilities(bundle['models'][model].predict_proba(x)[:, 1])
    ids = [r['record_id'] for r in rows]
    chosen = exact_capacity(p, ids, capacity)
    cutoff = bundle['cutoffs'][model]
    output = [{'record_id': r['record_id'], 'score': float(v), 'exact_capacity_review': bool(c),
               'development_cutoff_review': bool(v >= cutoff), 'cutoff_0_5_review': bool(v >= .5),
               **({'observed_churn': r['Churn']} if 'Churn' in r else {})} for r, v, c in zip(rows, p, chosen)]
    report = {'model': model, 'model_seed': 13, 'n': len(rows), 'capacity_fraction': capacity,
              'exact_capacity_count': int(chosen.sum()), 'development_cutoff': cutoff, 'rows': output,
              'interpretation': 'Historical benchmark model; no causal retention effect. Optional labels give descriptive errors, not independent validation.'}
    if 'Churn' in rows[0]:
        y = np.array([r['Churn'] for r in rows])
        report['descriptive_errors'] = confusion(y, chosen)
    return report


def export_csv(report):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
    writer.writeheader(); writer.writerows(report['rows'])
    return stream.getvalue()
