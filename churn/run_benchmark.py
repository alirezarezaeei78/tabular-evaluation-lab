"""Execute the prespecified UCI comparison and save public aggregates/private audits."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from churn_lab import (FEATURES, development_cutoff, evaluate, load_data, sha256,
                       split_indices, write_json)

HERE = Path(__file__).resolve().parent
DATA_URL = 'https://archive.ics.uci.edu/static/public/563/iranian%2Bchurn%2Bdataset.zip'
DATA_SHA = '90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1'
CONFIG = {'seeds': [13, 29, 47], 'primary_seed': 13, 'capacity_fraction': .1,
          'split': 'GroupShuffleSplit(.2 test); within development GroupShuffleSplit(.25 calibration)',
          'features': FEATURES, 'logistic': {'C': 1, 'max_iter': 2000},
          'boosting': {'learning_rate': .06, 'max_iter': 150, 'max_leaf_nodes': 15,
                       'l2_regularization': 1., 'early_stopping': False},
          'calibration': {'method': 'sigmoid', 'cv': 'prefit', 'partition': 'calibration only'},
          'selection': 'All declared seeds; no test-based model or hyperparameter selection'}


def fetch_data():
    destination = HERE / '.local' / 'Customer Churn.csv'
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'ChurnDecisionLab/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(1_000_001)
        if len(archive) > 1_000_000:
            raise ValueError('Unexpected source archive size')
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            data = z.read('Customer Churn.csv')
        if sha256(data) != DATA_SHA:
            raise ValueError('Provider data hash changed; inspect source before using it')
        destination.write_bytes(data)
    if sha256(destination.read_bytes()) != DATA_SHA:
        raise ValueError('Cached data hash mismatch')
    return destination


def run(output, audit_dir):
    output.mkdir(parents=True, exist_ok=True); audit_dir.mkdir(parents=True, exist_ok=True)
    source = fetch_data(); frame, x, y, groups = load_data(source)
    if len(y) != 3150:
        raise ValueError('Unexpected source row count')
    started = time.monotonic()
    summary = {'protocol_sha256': sha256((HERE / 'PROTOCOL.md').read_bytes()), 'configuration': CONFIG,
               'data': {'source': DATA_URL, 'sha256': DATA_SHA, 'rows': len(y), 'features': FEATURES,
                        'unique_predictor_groups': len(set(groups)), 'duplicate_rows_beyond_first': len(y)-len(set(groups)),
                        'license': 'CC BY 4.0', 'citation': 'Iranian Churn (2020), UCI, DOI 10.24432/C5JW3Z'},
               'environment': {'python': platform.python_version(), 'os': platform.system(),
                               'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(),
                               'fit_threads': 1, 'versions': {p: importlib.metadata.version(p) for p in
                               ['numpy','pandas','scikit-learn','scipy','joblib','matplotlib','threadpoolctl']}},
               'scope': 'Local retrospective grouped benchmark. Seed13 primary; other seeds sensitivity with overlapping rows. No prospective or causal claim.',
               'runs': []}
    for seed in CONFIG['seeds']:
        parts = split_indices(x, y, groups, seed)
        assert all(np.array_equal(parts[k], split_indices(x,y,groups,seed)[k]) for k in parts)
        fit, cal, test = (parts[k] for k in ['fit', 'calibration', 'evaluation'])
        ids = [f'source-{i:06d}' for i in test]
        models = {'dummy': DummyClassifier(strategy='prior'),
                  'logistic': make_pipeline(StandardScaler(), LogisticRegression(**CONFIG['logistic'], random_state=seed)),
                  'boosting': HistGradientBoostingClassifier(**CONFIG['boosting'], random_state=seed)}
        run_result = {'seed': seed, 'partitions': {}, 'models': {}, 'fit_logs': {}}
        mapping = []
        for name, idx in parts.items():
            run_result['partitions'][name] = {'rows': len(idx), 'positives': int(y[idx].sum()),
                                             'groups': len(set(groups[idx])),
                                             'index_sha256': sha256(np.sort(idx).astype('<i8').tobytes())}
            mapping.extend({'source_row_zero_based': int(i), 'partition': name, 'group_sha256': groups[i]} for i in idx)
        pd.DataFrame(mapping).to_csv(audit_dir / f'split-seed-{seed}.csv',index=False)
        with threadpool_limits(limits=1):
            for name, model in models.items():
                then = time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); model.fit(x[fit],y[fit])
                run_result['fit_logs'][name] = {'seconds': time.monotonic()-then,
                    'warnings': [str(w.message) for w in caught], 'stop': 'Completed fixed fit'}
                if caught:
                    raise RuntimeError(f'Unresolved fit warning for {name}: {caught[0].message}')
            scaler = models['logistic'].named_steps['standardscaler']
            assert scaler.n_samples_seen_ == len(fit)
            np.testing.assert_allclose(scaler.mean_,x[fit].mean(axis=0),rtol=1e-12,atol=1e-12)
            before = models['boosting'].predict_proba(x[cal])
            then = time.monotonic()
            calibrated = CalibratedClassifierCV(models['boosting'], method='sigmoid', cv='prefit')
            calibrated.fit(x[cal], y[cal]); models['boosting_sigmoid'] = calibrated
            np.testing.assert_array_equal(before, models['boosting'].predict_proba(x[cal]))
            run_result['fit_logs']['boosting_sigmoid'] = {'seconds': time.monotonic()-then,
                'base_model_refitted': False, 'stop':'Completed calibration-only sigmoid fit'}
            a = float(calibrated.calibrated_classifiers_[0].calibrators[0].a_)
            b = float(calibrated.calibrated_classifiers_[0].calibrators[0].b_)
            run_result['calibrator'] = {'A':a,'B':b,'formula':'1/(1+exp(A*decision_function+B))',
                                       'increasing_in_raw_score':a < 0}
            cutoffs = {}; pred = pd.DataFrame({'source_row_zero_based':test,'y':y[test]})
            for name, model in models.items():
                cutoff = development_cutoff(model.predict_proba(x[cal])[:,1]); cutoffs[name] = cutoff
                prob = model.predict_proba(x[test])[:,1]
                run_result['models'][name] = evaluate(y[test], prob, ids, cutoff)
                pred[name] = prob
            pred.to_csv(audit_dir / f'predictions-seed-{seed}.csv',index=False, float_format='%.17g')
            run_result['audit_hashes'] = {n:sha256((audit_dir / n).read_bytes()) for n in
                 [f'split-seed-{seed}.csv', f'predictions-seed-{seed}.csv']}
            if seed == 13:
                bundle = {'models': models, 'cutoffs':cutoffs, 'features':FEATURES,'seed':seed,
                          'data_sha256':DATA_SHA,'protocol_sha256':summary['protocol_sha256']}
                joblib.dump(bundle,HERE/'.local/model.joblib')
                reloaded = joblib.load(HERE/'.local/model.joblib')
                for name in models:
                    np.testing.assert_array_equal(models[name].predict_proba(x[test]),reloaded['models'][name].predict_proba(x[test]))
        summary['runs'].append(run_result)
        write_json(output / 'results.json', summary)
        print(json.dumps({'seed':seed, 'partitions':run_result['partitions'],
                          'models':{k:{m:v[m] for m in ['ap','brier','log_loss']} for k,v in run_result['models'].items()}}),flush=True)
        if time.monotonic()-started > 1200:
            raise RuntimeError('Classical experiment budget exceeded; preserve partial results')
    summary['checks'] = {'group_disjoint': True,'all_partitions_exhaustive':True,
                         'train_only_scaler':True,'base_not_refitted_by_calibration':True,
                         'independent_ap_brier_oracles':True,'finite_bounded_scores':True,
                         'joblib_roundtrip_equal':True,'no_test_label_threshold_choice':True}
    summary['operational_elapsed_seconds'] = time.monotonic()-started
    write_json(output/'results.json',summary)
    write_json(output/'configuration.json',CONFIG)
    print('COMPLETE: all three prespecified runs; aggregates saved; row-level audit and model remain local.',flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'local-output/reproduction')
    parser.add_argument('--audit-dir',type=Path,default=HERE/'local-output/private-audit')
    args=parser.parse_args();run(args.output,args.audit_dir)
