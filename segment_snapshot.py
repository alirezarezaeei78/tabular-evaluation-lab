"""Compare RFM snapshots in one frozen model namespace; local JSON/CSV only."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score
from retail_rfm import exact_ari

FIELDS = ['CustomerID', 'R', 'F', 'M']
METHODS = ['kmeans', 'median_rule']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded(path, limit=5_000_000):
    with Path(path).open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Input exceeds byte limit')
    return data


def snapshot(path):
    data = bounded(path)
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != FIELDS:
        raise ValueError('CSV header must be CustomerID,R,F,M in that order')
    rows = {}
    for row in reader:
        if len(rows) >= 50000 or set(row) != set(FIELDS) or any(v is None for v in row.values()):
            raise ValueError('Malformed row or row limit exceeded')
        key = row['CustomerID']
        if not key.strip() or key != key.strip() or len(key) > 128 or key in rows:
            raise ValueError('Require unique nonblank IDs without surrounding whitespace, at most128 characters')
        x = np.array([float(row[f]) for f in FIELDS[1:]])
        if not np.isfinite(x).all() or x[0] < 0 or x[1] < 1 or x[1] % 1 or x[2] <= 0:
            raise ValueError('Require finite R>=0, integer F>=1 and M>0')
        rows[key] = x
    return rows, digest(data)


def model(path):
    data = bounded(path, 100000)
    obj = json.loads(data)
    if not isinstance(obj, dict) or obj.get('schema') != 'frozen-rfm-v1' or obj.get('method') != 'log_standard':
        raise ValueError('Require frozen-rfm-v1 log_standard model')
    for name, shape in [('center', (3,)), ('scale', (3,)), ('centers', (2, 3)), ('development_medians', (3,))]:
        x = np.asarray(obj[name], dtype=float)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError('Invalid model shape or nonfinite parameters')
    if (np.array(obj['scale']) <= 0).any() or (np.array(obj['development_medians']) < 0).any():
        raise ValueError('Invalid model scale or medians')
    return obj, digest(data)


def predict(obj, rows):
    keys = sorted(rows)
    if not keys:
        return {method: {} for method in METHODS}
    x = np.array([rows[k] for k in keys])
    z = (np.log1p(x) - np.asarray(obj['center'])) / np.asarray(obj['scale'])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        distances = ((z[:, None, :] - np.asarray(obj['centers'])[None, :, :]) ** 2).sum(axis=2)
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite distance')
    learned = distances.argmin(axis=1)  # Equal distances choose the lower label.
    med = np.asarray(obj['development_medians'])
    votes = np.column_stack([x[:, 0] <= med[0], x[:, 1] >= med[1], x[:, 2] >= med[2]])
    rule = (votes.sum(axis=1) >= 2).astype(int)
    return {method: dict(zip(keys, labels.tolist())) for method, labels in zip(METHODS, [learned, rule])}


def compare(left_path, right_path, model_path):
    left, lh = snapshot(left_path)
    right, rh = snapshot(right_path)
    obj, mh = model(model_path)
    assignments = {side: predict(obj, rows) for side, rows in [('left', left), ('right', right)]}
    common = sorted(left.keys() & right.keys())
    n = len(common)
    report = {'schema': 'snapshot-comparison-v1', 'model_sha256': mh,
              'input_sha256': {'left': lh, 'right': rh}, 'frozen_model': True,
              'accounting': {'left': len(left), 'right': len(right), 'matched': n,
                             'left_only': len(left.keys() - right.keys()),
                             'right_only': len(right.keys() - left.keys()),
                             'union': len(left.keys() | right.keys()),
                             'left_matched_share': n / len(left) if left else None,
                             'right_matched_share': n / len(right) if right else None},
              'methods': {}, 'profiles': []}
    for method in METHODS:
        a = [assignments['left'][method][k] for k in common]
        b = [assignments['right'][method][k] for k in common]
        changed = sum(x != y for x, y in zip(a, b))
        ari = float(adjusted_rand_score(a, b)) if n >= 2 else None
        if n >= 2 and abs(ari - exact_ari(a, b)) > 1e-12:
            raise AssertionError('Independent ARI mismatch')
        counts = Counter(zip(a, b))
        report['methods'][method] = {'matched': n, 'changed_fixed_labels': changed,
            'changed_fixed_label_share': changed / n if n else None, 'ari': ari,
            'metric_status': 'ok' if n >= 2 else 'fewer_than_two_matches',
            'transitions': [{'left_label': i, 'right_label': j, 'count': counts[i, j]}
                            for i in range(2) for j in range(2)]}
        for side, rows in [('left', left), ('right', right)]:
            for scope, ids in [('all', sorted(rows)), ('matched', common)]:
                for label in range(2):
                    values = [rows[k] for k in ids if assignments[side][method][k] == label]
                    medians = np.median(values, axis=0).tolist() if len(values) >= 10 else [None] * 3
                    report['profiles'].append(dict(method=method, side=side, scope=scope,
                        label=label, count=len(values), median_R=medians[0], median_F=medians[1],
                        median_M=medians[2], medians_suppressed=len(values) < 10))
    records = []
    for key in sorted(left.keys() | right.keys()):
        row = {'CustomerID': key, 'status': 'matched' if key in left and key in right else 'left_only' if key in left else 'right_only'}
        for side, rows in [('left', left), ('right', right)]:
            for method in METHODS:
                row[side + '_' + method] = assignments[side][method].get(key)
        records.append(row)
    return report, records


def export(report, records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous report/input.
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    def write(name, rows, fields):
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    write('assignments.csv', records, ['CustomerID', 'status', 'left_kmeans', 'left_median_rule', 'right_kmeans', 'right_median_rule'])
    transitions = [dict(method=m, **r) for m in METHODS for r in report['methods'][m]['transitions']]
    write('transitions.csv', transitions, ['method', 'left_label', 'right_label', 'count'])
    write('profiles.csv', report['profiles'], ['method', 'side', 'scope', 'label', 'count', 'median_R', 'median_F', 'median_M', 'medians_suppressed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['left', 'right', 'model', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        report, rows = compare(args.left, args.right, args.model)
        export(report, rows, args.output)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, FloatingPointError, csv.Error) as exc:
        print('Input/output error: ' + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({'accounting': report['accounting'], 'methods': report['methods']}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
