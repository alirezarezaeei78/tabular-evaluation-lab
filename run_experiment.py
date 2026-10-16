"""Execute correctness checks and all preset split-auditing fixtures."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

import test_split_audit
from split_audit import audit, read_csv, write_report

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_bytes((json.dumps(obj, indent=2) + '\n').encode())


def execute(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        raise ValueError('Results already exist; choose a new output directory.')
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_split_audit)
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'tests.log').write_bytes(log.getvalue().encode())
    if not tests.wasSuccessful():
        raise RuntimeError('Correctness checks failed; no publishable experiment.')
    declaration = json.loads((ROOT/'declaration.json').read_text())
    assert sha(ROOT/'PROTOCOL.md') == declaration['protocol_sha256']
    assert sha(ROOT/'config.json') == declaration['config_sha256']
    fixture_manifest = json.loads((ROOT/'fixtures/fixture-manifest.json').read_text())
    for name, expected in fixture_manifest['files'].items():
        assert sha(ROOT/'fixtures'/name) == expected, name
    cases = json.loads((ROOT/'fixtures/cases.json').read_text())
    summaries = []
    for case in cases:
        train_path = ROOT/'fixtures'/case['train_file']
        test_path = ROOT/'fixtures'/case['test_file']
        train, test = read_csv(train_path), read_csv(test_path)
        if case['method'] != 'independent_common_values':
            train_ids = [r['row_id'] for r in train]
            test_ids = [r['row_id'] for r in test]
            assert len(train_ids) + len(test_ids) == 600
            assert len(set(train_ids + test_ids)) == 600
            assert set(train_ids).isdisjoint(test_ids)
        before = copy.deepcopy((train, test))
        report = audit(train, test, ['measurement', 'channel'], numeric_columns=['measurement'],
                       group_column='entity', target_column='target')
        repeated = audit(train, test, ['measurement', 'channel'], numeric_columns=['measurement'],
                         group_column='entity', target_column='target')
        assert repeated == report
        assert (train, test) == before
        actual = {key: report['counts'][key] for key in case['expected']}
        assert actual == case['expected'], (case['id'], actual, case['expected'])
        report['input_files'] = {split: {'sha256': sha(path), 'bytes': path.stat().st_size}
                                 for split, path in [('train', train_path), ('test', test_path)]}
        write_report(output/case['id'], report)
        # Separate baseline: no audit produces no warnings, missing every known intersection.
        summaries.append({'case': case['id'], 'method': case['method'], 'seed': case['seed'],
                          'train_rows': len(train), 'test_rows': len(test), **actual,
                          'oracle_counts_match': True, 'all_input_rows_preserved': True,
                          'unchecked_baseline_feature_rows_flagged': 0,
                          'unchecked_baseline_group_rows_flagged': 0,
                          'repeated_report_identical': True})
    result = {'executed_at': datetime.now(timezone.utc).isoformat(),
              'classification': 'Authored synthetic functionality/stress fixtures; no independent validation',
              'checks_passed': tests.testsRun-len(tests.skipped), 'checks_skipped': len(tests.skipped),
              'cases_executed': len(cases), 'case_counts_match_independent_oracle': len(summaries),
              'model_fits': 0, 'runtime_comparison': False, 'summaries': summaries,
              'parameters': json.loads((ROOT/'config.json').read_text()),
              'environment': {'python': platform.python_version(), 'platform': platform.system(),
                              'core_dependencies': 'Python standard library only'},
              'input_manifest_sha256': sha(ROOT/'fixtures/fixture-manifest.json'),
              'source_sha256': {n: sha(ROOT/n) for n in ['split_audit.py', 'test_split_audit.py', 'generate_fixtures.py', 'run_experiment.py', 'PROTOCOL.md', 'config.json']},
              'command': 'python run_experiment.py --output ' + str(output).replace('\\', '/'),
              'limits': ['Different partition methods have different held-out rows/sizes; no matched predictive comparison.',
                         'Selected-feature equality can be legitimate across distinct entities.',
                         'No near-duplicate, chronology, target-proxy or unrecorded entity detection.',
                         'Tiny synthetic fixture family visible during development; not enterprise validation.']}
    save(output/'results.json', result)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(summaries)
    (output/'summary.csv').write_bytes(stream.getvalue().encode())
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/2026-10-16')
    execute(parser.parse_args().output)
