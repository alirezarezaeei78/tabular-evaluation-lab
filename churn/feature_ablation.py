"""Frozen matched-row refitting ablation; public aggregates, local model audit."""
import argparse
import csv
import importlib.metadata
import io
import json
import math
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from churn_lab import FEATURES, evaluate, load_data, sha256, split_indices, write_json
from run_benchmark import CONFIG, DATA_SHA, fetch_data

HERE = Path(__file__).resolve().parent
OUT = HERE / 'experiments/2026-10-29'
GROUPS = {
    'quality': ['Call Failure', 'Complains'],
    'usage': ['Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers'],
    'subscription': ['Subscription Length', 'Charge Amount', 'Tariff Plan'],
    'age': ['Age Group', 'Age'],
}


def choose_feature(x, y):
    """Point-biserial correlation on FITTING rows only; input order breaks ties."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError('Aligned nonempty fitting inputs required')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or set(y) != {0, 1}:
        raise ValueError('Finite fitting features and both binary classes required')
    xc = x - x.mean(axis=0); yc = y - y.mean()
    denominator = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
    corr = np.divide((xc * yc[:, None]).sum(axis=0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)
    return int(np.argmax(np.abs(corr))), corr


def variants(single):
    out = {'full': list(range(len(FEATURES)))}
    for name, group in GROUPS.items():
        out['without_' + name] = [i for i, f in enumerate(FEATURES) if f not in group]
    out['single_feature'] = [single]
    return out


def equivalent_rows(xfit, xtest):
    """Count evaluation feature tuples exactly present among fitting tuples."""
    known = set(map(tuple, np.asarray(xfit).tolist()))
    return sum(tuple(row) in known for row in np.asarray(xtest).tolist())


def export_slice(report, seed, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Export already exists; choose a new path')
    if path.suffix.lower() not in ['.json', '.csv']:
        raise ValueError('Export must be CSV or JSON')
    rows = report['summary_rows']
    if seed is not None:
        rows = [row for row in rows if row['seed'] == seed]
    if not rows:
        raise ValueError('No measured seed selected')
    if path.suffix.lower() == '.json':
        write_json(path, rows)
    else:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
        path.write_text(stream.getvalue(), encoding='utf-8', newline='\n')
    return rows


def run(output, audit):
    output = Path(output); audit = Path(audit)
    if (output / 'results.json').exists() or audit.exists():
        raise FileExistsError('Preserve previous evidence; use new output/audit directories')
    frozen = json.loads((OUT / 'protocol-frozen.json').read_text(encoding='utf-8'))
    assert sha256((OUT / 'PROTOCOL.md').read_bytes()) == frozen['protocol_sha256']
    output.mkdir(parents=True, exist_ok=True); audit.mkdir(parents=True)
    started = time.monotonic()
    source = fetch_data(); assert sha256(source.read_bytes()) == DATA_SHA
    _, x, y, groups = load_data(source)
    old = json.loads((HERE / 'experiments/2026-10-21/results.json').read_text(encoding='utf-8'))
    result = {'status': 'running', 'scope': 'Exploratory reuse of 21/25 October grouped evaluations',
              'protocol_sha256': frozen['protocol_sha256'], 'frozen_at': frozen['frozen_at'],
              'source_sha256': DATA_SHA, 'groups': GROUPS, 'features': FEATURES,
              'parameters': CONFIG['boosting'], 'primary_seed': 13,
              'source_citation': 'Iranian Churn (2020), UCI, DOI 10.24432/C5JW3Z, CC BY 4.0',
              'environment': {'python': platform.python_version(), 'cpu': platform.processor(),
                              'fit_threads': 1, 'versions': {p: importlib.metadata.version(p) for p in
                              ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']}},
              'runs': [], 'summary_rows': []}
    with threadpool_limits(limits=1):
        for seed in [13, 29, 47]:
            parts = split_indices(x, y, groups, seed); fit = parts['fit']; test = parts['evaluation']
            reference = next(r for r in old['runs'] if r['seed'] == seed)
            partition_records = {k: {'rows': len(v), 'positives': int(y[v].sum()),
                                    'index_sha256': sha256(np.sort(v).astype('<i8').tobytes())}
                                 for k, v in parts.items()}
            for k in parts:
                assert partition_records[k]['index_sha256'] == reference['partitions'][k]['index_sha256']
            single, corr = choose_feature(x[fit], y[fit]); columns = variants(single)
            models = {}; logs = {}
            # No evaluation labels are used until all declared fits finish.
            for name, cols in columns.items():
                model = HistGradientBoostingClassifier(**CONFIG['boosting'], random_state=seed)
                then = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); model.fit(x[fit][:, cols], y[fit])
                if caught:
                    raise RuntimeError(f'{name}: {caught[0].message}')
                assert model.n_iter_ == 150
                logs[name] = {'seconds': time.monotonic() - then, 'iterations': int(model.n_iter_),
                              'warnings': [], 'stop': 'completed fixed 150 iterations'}
                joblib.dump({'model': model, 'columns': cols}, audit / f'{seed}-{name}.joblib')
                models[name] = model
                if time.monotonic() - started > 1200:
                    raise RuntimeError('20-minute total budget exceeded; preserve partial audit')
            ids = [f'source-{i:06d}' for i in test]
            predictions = pd.DataFrame({'source_row_zero_based': test, 'y': y[test]})
            metrics = {}; collisions = {}; replay = {}
            for name in ['prior'] + list(columns):
                if name == 'prior':
                    prob = np.full(len(test), y[fit].mean()); cols = []
                else:
                    cols = columns[name]; prob = models[name].predict_proba(x[test][:, cols])[:, 1]
                    loaded = joblib.load(audit / f'{seed}-{name}.joblib')
                    np.testing.assert_array_equal(prob, loaded['model'].predict_proba(x[test][:, loaded['columns']])[:, 1])
                    replay[name] = True
                metrics[name] = evaluate(y[test], prob, ids, .5)
                order = sorted(range(len(prob)), key=lambda i: (-float(prob[i]), ids[i]))
                k = math.ceil(len(prob) / 10)
                assert metrics[name]['policies']['exact_10_percent']['tp'] == sum(int(y[test][i]) for i in order[:k])
                collisions[name] = equivalent_rows(x[fit][:, cols], x[test][:, cols]) if cols else None
                predictions[name] = prob
            for metric in ['ap', 'brier']:
                assert abs(metrics['full'][metric] - reference['models']['boosting'][metric]) < 1e-12
            mapping = [{'source_row_zero_based': int(i), 'partition': k, 'full_group_hash': groups[i]}
                       for k, indices in parts.items() for i in indices]
            pd.DataFrame(mapping).to_csv(audit / f'split-{seed}.csv', index=False)
            predpath = audit / f'predictions-{seed}.csv'
            predictions.to_csv(predpath, index=False, float_format='%.17g')
            replay_csv = pd.read_csv(predpath, float_precision='round_trip')
            for name in metrics:
                np.testing.assert_array_equal(replay_csv[name].to_numpy(), predictions[name].to_numpy())
                p = metrics[name]['policies']['exact_10_percent']
                result['summary_rows'].append({'seed': seed, 'variant': name,
                    'n_features': len(columns.get(name, [])), 'evaluation_n': len(test),
                    'evaluation_positives': int(y[test].sum()), 'ap': metrics[name]['ap'],
                    'delta_ap_vs_full': metrics[name]['ap'] - metrics['full']['ap'],
                    'brier': metrics[name]['brier'],
                    'delta_brier_vs_full': metrics[name]['brier'] - metrics['full']['brier'],
                    'selected': p['selected'], 'tp': p['tp'], 'fp': p['fp'], 'fn': p['fn'],
                    'feature_equivalent_fit_rows': collisions[name]})
            one = {'seed': seed, 'partitions': partition_records,
                   'single_feature': FEATURES[single], 'fit_only_correlations': dict(zip(FEATURES, corr.tolist())),
                   'columns': {n: [FEATURES[i] for i in v] for n, v in columns.items()},
                   'metrics': metrics, 'fit_logs': logs, 'model_replay_exact': replay,
                   'reduced_feature_equivalent_evaluation_rows': collisions}
            result['runs'].append(one)
            write_json(output / 'partial-results.json', result)
            print(json.dumps({'seed': seed, 'single_feature': FEATURES[single],
                              'results': [r for r in result['summary_rows'] if r['seed'] == seed]}), flush=True)
    result['status'] = 'complete'
    result['checks'] = {'original_split_hashes_match': True, 'full_model_reproduced': True,
                        'independent_ap_brier_capacity': True, 'all_18_model_replays_exact': True,
                        'all_3_prediction_csv_roundtrips_exact': True, 'all_variants_retained': True}
    result['audit_hashes'] = {p.name: sha256(p.read_bytes()) for p in sorted(audit.iterdir()) if p.is_file()}
    result['operational_elapsed_seconds'] = time.monotonic() - started
    write_json(output / 'results.json', result)
    export_slice(result, None, output / 'summary.csv')
    print('COMPLETE: 18 fixed fits; all 21 measured rows saved; no model selected from exposed tests.', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser('run'); runner.add_argument('--output', type=Path, default=OUT)
    runner.add_argument('--audit-dir', type=Path, default=HERE / '.local/feature-ablation-2026-10-29')
    inspect = commands.add_parser('inspect'); inspect.add_argument('--report', type=Path, default=OUT / 'results.json')
    inspect.add_argument('--seed', type=int, choices=[13, 29, 47]); inspect.add_argument('--export', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run': run(args.output, args.audit_dir)
    else: print(json.dumps(export_slice(json.loads(args.report.read_text(encoding='utf-8')), args.seed, args.export), indent=2))
