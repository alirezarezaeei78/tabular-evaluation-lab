"""Correctness, exact-equivalence, collision and real CLI workflow checks."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from split_audit import audit, canonical_key, flags_csv, numeric, read_csv, read_csv_snapshot, write_report

ROOT = Path(__file__).resolve().parent


def row(value='1', group='a', label='0', text='zone'):
    return {'x': value, 'kind': text, 'entity': group, 'target': label}


def check(train, test, **kwargs):
    return audit(train, test, ['x', 'kind'], numeric_columns=['x'],
                 group_column='entity', target_column='target', **kwargs)


class EqualityChecks(unittest.TestCase):
    def test_numeric_notation_equivalence(self):
        values = ['1', '1.0', '1e0', ' 01.000 ', '10e-1']
        self.assertEqual(len({json.dumps(numeric(v)) for v in values}), 1)

    def test_signed_zero(self):
        self.assertEqual(numeric('-0.0'), numeric('0e200'))

    def test_large_precision_not_rounded(self):
        a = '123456789012345678901234567890.1'
        b = '123456789012345678901234567890.2'
        self.assertNotEqual(numeric(a), numeric(b))

    def test_close_values_not_rounded(self):
        self.assertNotEqual(numeric('1.0000000000000000000000000001'), numeric('1'))

    def test_nonfinite_rejected(self):
        for value in ['NaN', 'Infinity', '-Infinity', 'sNaN']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                numeric(value)

    def test_bad_numeric_rejected(self):
        for value in ['fish', '1,000', '1.2.3', '7'*257]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                numeric(value)

    def test_unicode_nfc_equivalence(self):
        result = check([row(text='caf\u00e9')], [row(text='cafe\u0301')])
        self.assertEqual(result['counts']['overlapping_test_rows'], 1)

    def test_text_case_preserved(self):
        self.assertEqual(check([row(text='Zone')], [row(text='zone')])['counts']['shared_feature_keys'], 0)

    def test_text_whitespace_preserved(self):
        self.assertEqual(check([row(text='zone')], [row(text=' zone')])['counts']['shared_feature_keys'], 0)

    def test_missing_is_not_literal_text(self):
        a = canonical_key({'x': ''}, ('x',), frozenset())
        b = canonical_key({'x': 'missing'}, ('x',), frozenset())
        self.assertNotEqual(a, b)

    def test_empty_numeric_whitespace_is_missing(self):
        self.assertEqual(numeric(''), numeric(' \t'))

    def test_column_order_invariance(self):
        a = audit([row()], [row()], ['kind', 'x'], numeric_columns=['x'])
        b = audit([dict(reversed(list(row().items())))], [row()], ['x', 'kind'], numeric_columns=['x'])
        self.assertEqual(a, b)

    def test_targets_do_not_influence_feature_identity(self):
        result = check([row(label='0')], [row(label='1')])
        self.assertEqual(result['counts']['shared_feature_keys'], 1)

    def test_duplicate_multiplicity_and_positions(self):
        result = check([row(), row()], [row(), row(), row('2', 'b')])
        self.assertEqual(result['counts']['shared_feature_keys'], 1)
        self.assertEqual(result['counts']['overlapping_test_rows'], 2)
        self.assertEqual(result['feature_matches'][0]['train_rows'], [1, 2])
        self.assertEqual(result['feature_matches'][0]['test_rows'], [1, 2])

    def test_group_overlap_without_feature_match(self):
        result = check([row('1')], [row('2')])
        self.assertEqual(result['counts']['shared_feature_keys'], 0)
        self.assertEqual(result['counts']['shared_groups'], 1)

    def test_feature_overlap_without_group_match(self):
        result = check([row('1', 'a')], [row('1', 'b')])
        self.assertEqual(result['counts']['shared_feature_keys'], 1)
        self.assertEqual(result['counts']['shared_groups'], 0)
        self.assertIn('not proof of leakage', result['limits'][0])

    def test_no_group_is_not_zero_groups(self):
        result = audit([row()], [row('2')], ['x'], numeric_columns=['x'])
        self.assertIsNone(result['counts']['shared_groups'])
        self.assertIsNone(result['test_row_flags'][0]['group_seen_in_train'])

    def test_blank_group_rejected(self):
        for value in ['', '  ']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                check([row(group=value)], [row()])

    def test_all_missing_features_counted_and_warned(self):
        result = audit([{'x': ''}], [{'x': ''}], ['x'])
        self.assertEqual(result['counts']['all_features_missing'], {'train': 1, 'test': 1})
        self.assertTrue(any('all_features_missing' in w for w in result['warnings']))

    def test_no_input_mutation(self):
        train, test = [row()], [row('2')]
        before = copy.deepcopy((train, test))
        check(train, test)
        self.assertEqual((train, test), before)

    def test_deterministic_report(self):
        self.assertEqual(check([row()], [row()]), check([row()], [row()]))

    def test_digest_collision_cannot_create_false_match(self):
        with patch('split_audit.digest', return_value='forced-collision'):
            result = check([row('1', 'a')], [row('2', 'b')])
        self.assertEqual(result['counts']['shared_feature_keys'], 0)
        self.assertEqual(result['counts']['shared_groups'], 0)
        self.assertEqual(result['counts']['digest_collision_buckets'], 1)

    def test_report_does_not_expose_raw_cells(self):
        result = check([row('12', 'private-entity', text='private-feature')],
                       [row('12', 'private-entity', text='private-feature')])
        self.assertNotIn('private-entity', json.dumps(result))
        self.assertNotIn('private-feature', json.dumps(result))

    def test_json_and_csv_parity(self):
        result = check([row()], [row(), row('2', 'b')])
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
        table = list(csv.DictReader(io.StringIO(flags_csv(result))))
        self.assertEqual(len(table), 2)
        self.assertEqual([r['selected_features_seen_in_train'] for r in table], ['True', 'False'])


class ValidationChecks(unittest.TestCase):
    def test_empty_data(self):
        for train, test in [([], [row()]), ([row()], [])]:
            with self.subTest(train=train), self.assertRaises(ValueError):
                check(train, test)

    def test_bad_features(self):
        for features in [[], ['x', 'x'], 'x', [True], ['']]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                audit([row()], [row()], features)

    def test_numeric_subset_required(self):
        with self.assertRaises(ValueError):
            audit([row()], [row()], ['x'], numeric_columns=['target'])

    def test_target_cannot_be_a_feature(self):
        with self.assertRaises(ValueError):
            audit([row()], [row()], ['x', 'target'], target_column='target')

    def test_role_names_and_conflicts(self):
        for kwargs in [{'group_column': ''}, {'group_column': True},
                       {'group_column': 'entity', 'target_column': 'entity'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit([row()], [row()], ['x'], **kwargs)

    def test_missing_required_columns(self):
        with self.assertRaises(ValueError):
            check([row()], [{'x': '1', 'entity': 'a', 'kind': 'zone'}])

    def test_wrong_cell_type_and_oversized_text(self):
        for value in [None, 1, True, 'x'*4097]:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                check([row(value)], [row()])

    def test_invalid_row_schema(self):
        for rows in [{}, [None], [{}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                check(rows, [row()])


class FileAndCLIChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def file(self, content, name='data.csv'):
        path = self.directory/name
        path.write_bytes(content.encode('utf-8'))
        return path

    def test_csv_multiline_bom_and_quoted_values(self):
        path = self.file('\ufeffx,kind,entity,target\n1,"zone,\nalpha",a,0\n')
        self.assertEqual(read_csv(path)[0]['kind'], 'zone,\nalpha')

    def test_header_only_empty_and_duplicate_header_rejected(self):
        for content in ['', 'x,kind\n', 'x,x\n1,2\n']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(self.file(content))

    def test_ragged_csv_rejected(self):
        for content in ['x,kind\n1\n', 'x,kind\n1,2,3\n', 'x\n"unfinished\n']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(self.file(content))

    def test_invalid_utf8_rejected(self):
        path = self.directory/'invalid.csv'
        path.write_bytes(b'x\n\xff\n')
        with self.assertRaises(ValueError):
            read_csv(path)

    def test_parsed_snapshot_hash_matches_exact_bytes(self):
        raw = b'\xef\xbb\xbfx,kind,entity,target\r\n1,zone,a,0\r\n'
        path = self.directory/'snapshot.csv'
        path.write_bytes(raw)
        rows, metadata = read_csv_snapshot(path)
        self.assertEqual(rows[0], row())
        self.assertEqual(metadata, {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})

    def test_input_size_limit(self):
        path = self.file('x\n' + 'a'*5_000_001)
        with self.assertRaises(ValueError):
            read_csv(path)

    def test_row_limit_rejected_without_truncation(self):
        path = self.file('x\n' + 'a\n'*50_001)
        with self.assertRaises(ValueError):
            read_csv(path)

    def test_report_no_overwrite_without_flag(self):
        result = check([row()], [row()])
        write_report(self.directory, result)
        with self.assertRaises(ValueError):
            write_report(self.directory, result)
        write_report(self.directory, result, overwrite=True)
        self.assertEqual(json.loads((self.directory/'report.json').read_text()), result)

    def test_input_cannot_be_report_target(self):
        with self.assertRaises(ValueError):
            write_report(self.directory, check([row()], [row()]),
                         input_paths=[self.directory/'report.json'], overwrite=True)

    def run_cli(self, train, test):
        def write(name, rows):
            path = self.directory/name
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row()), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            return path
        a, b = write('train.csv', train), write('test.csv', test)
        before = (a.read_bytes(), b.read_bytes())
        command = [sys.executable, '-X', 'utf8', str(ROOT/'split_audit.py'),
                   '--train', str(a), '--test', str(b), '--features', 'x', 'kind',
                   '--numeric', 'x', '--group', 'entity', '--target', 'target',
                   '--output', str(self.directory/'output'), '--fail-on-overlap']
        process = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual((a.read_bytes(), b.read_bytes()), before)
        return process

    def test_cli_clean_exit_and_exports(self):
        process = self.run_cli([row('1', 'a')], [row('2', 'b')])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)['status'], 'no_overlap_found')
        self.assertTrue((self.directory/'output/test-row-flags.csv').is_file())

    def test_cli_overlap_exit_and_evidence(self):
        process = self.run_cli([row()], [row()])
        self.assertEqual(process.returncode, 1, process.stderr)
        report = json.loads((self.directory/'output/report.json').read_text())
        self.assertEqual(report['feature_matches'][0]['test_rows'], [1])
        self.assertEqual(json.loads(process.stdout)['counts'], report['counts'])

    def test_cli_input_error_has_no_report(self):
        process = self.run_cli([row('NaN')], [row()])
        self.assertEqual(process.returncode, 2)
        self.assertEqual(json.loads(process.stderr)['status'], 'input_error')
        self.assertFalse((self.directory/'output/report.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
