"""Inspect exact feature and group overlap across local train/test CSV files.

No input rows are changed or removed. A warning is not proof of leakage; absence
of a warning is not a general validation certificate. Python standard library.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata

MAX_ROWS = 50_000
MAX_FILE_BYTES = 5_000_000
MAX_CELL_CHARS = 4096
MAX_COLUMNS = 128


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_CELL_CHARS:
        raise ValueError('Cells must be strings with at most 4096 characters.')
    return unicodedata.normalize('NFC', value)


def numeric(value: str) -> list:
    """Exact decimal identity, with no context-dependent rounding."""
    value = text(value).strip()
    if not value:
        return ['missing']
    if len(value) > 256:
        raise ValueError('Numeric cells are limited to 256 characters.')
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('A numeric feature contains an invalid decimal.') from exc
    if not number.is_finite():
        raise ValueError('Numeric features must be finite.')
    if number == 0:
        return ['number', 0, '0', 0]
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return ['number', sign, ''.join(map(str, digits)), exponent]


def canonical_key(row: dict, features: tuple, numeric_columns: frozenset) -> bytes:
    cells = []
    for name in sorted(features):
        value = text(row[name])
        token = numeric(value) if name in numeric_columns else (
            ['missing'] if value == '' else ['text', value])
        cells.append([name, token])
    return json.dumps(cells, ensure_ascii=True, separators=(',', ':')).encode('ascii')


def _validate_rows(rows: list, required: set, split: str) -> None:
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError(f'{split} needs 1 to {MAX_ROWS} rows.')
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not row or len(row) > MAX_COLUMNS:
            raise ValueError(f'{split} data row {index} has invalid columns.')
        if not required <= row.keys():
            raise ValueError(f'{split} data row {index} is missing required columns.')
        for name, value in row.items():
            if not isinstance(name, str) or not name or len(name) > 128:
                raise ValueError('Column names must be nonempty strings of at most 128 characters.')
            text(value)


def _index(keys: list[bytes]) -> dict:
    found = {}
    for position, key in enumerate(keys, 1):
        found.setdefault(key, []).append(position)
    return found


def _shared(train_keys: list[bytes], test_keys: list[bytes]) -> tuple:
    train_index, test_index = _index(train_keys), _index(test_keys)
    # Compare the complete canonical bytes, never digest equality alone.
    shared = sorted(train_index.keys() & test_index.keys())
    evidence = [{'fingerprint': digest(key), 'train_rows': train_index[key],
                 'test_rows': test_index[key]} for key in shared]
    flagged = {i for key in shared for i in test_index[key]}
    return evidence, flagged


def audit(train: list, test: list, features: list | tuple, *, numeric_columns=(),
          group_column: str | None = None, target_column: str | None = None) -> dict:
    if not isinstance(features, (list, tuple)) or not 1 <= len(features) <= 64:
        raise ValueError('Specify 1 to 64 feature columns explicitly.')
    if any(not isinstance(c, str) or not c or len(c) > 128 for c in features):
        raise ValueError('Feature names must be nonempty strings of at most 128 characters.')
    if len(set(features)) != len(features):
        raise ValueError('Feature names must be unique.')
    if not isinstance(numeric_columns, (list, tuple, set, frozenset)):
        raise ValueError('Numeric columns must be a collection of feature names.')
    if any(not isinstance(c, str) for c in numeric_columns):
        raise ValueError('Numeric column names must be strings.')
    numeric_columns = frozenset(numeric_columns)
    if not numeric_columns <= set(features):
        raise ValueError('Numeric columns must be selected features.')
    for name in (group_column, target_column):
        if name is not None and (not isinstance(name, str) or not name or len(name) > 128):
            raise ValueError('Group/target names must be nonempty strings.')
    if target_column in features:
        raise ValueError('Exclude the target from feature fingerprints.')
    if group_column is not None and group_column == target_column:
        raise ValueError('Group and target roles must use different columns.')
    required = set(features) | {c for c in (group_column, target_column) if c is not None}
    _validate_rows(train, required, 'train')
    _validate_rows(test, required, 'test')
    features = tuple(sorted(features))
    train_keys = [canonical_key(row, features, numeric_columns) for row in train]
    test_keys = [canonical_key(row, features, numeric_columns) for row in test]
    matches, feature_flagged = _shared(train_keys, test_keys)
    group_matches, group_flagged = None, set()
    if group_column is not None:
        def group_key(row):
            value = text(row[group_column])
            if not value.strip():
                raise ValueError('Group identifiers cannot be empty or whitespace-only.')
            return json.dumps(['group', value], ensure_ascii=True).encode('ascii')
        group_matches, group_flagged = _shared([group_key(r) for r in train],
                                               [group_key(r) for r in test])
    all_keys = set(train_keys + test_keys)
    digest_buckets = {}
    for key in all_keys:
        digest_buckets.setdefault(digest(key), set()).add(key)
    collisions = sum(len(values) > 1 for values in digest_buckets.values())
    missing_counts = {}
    for split, rows in [('train', train), ('test', test)]:
        missing_counts[split] = sum(all(
            (text(r[c]).strip() == '' if c in numeric_columns else text(r[c]) == '')
            for c in features) for r in rows)
    warnings = []
    if matches:
        warnings.append('selected_feature_overlap: inspect source identity and task context')
    if group_matches:
        warnings.append('shared_groups: review whether evaluation requires unseen entities')
    if group_column is None:
        warnings.append('group_check_not_run: no explicit group column supplied')
    if any(missing_counts.values()):
        warnings.append('all_features_missing: missing tuples can create uninformative matches')
    if collisions:
        warnings.append('digest_collision: canonical-byte equality used; digest labels may repeat')
    found = bool(matches or group_matches)
    return {
        'schema_version': 1,
        'status': 'overlap_detected' if found else 'no_overlap_found',
        'rows': {'train': len(train), 'test': len(test)},
        'features': list(features), 'numeric_columns': sorted(numeric_columns),
        'group_column': group_column, 'target_column_excluded': target_column,
        'policy': {'text': 'Unicode NFC; case and whitespace preserved',
                   'numeric': 'Finite exact Decimal; whitespace stripped; notation/signed-zero normalized',
                   'missing': 'Empty text cell or whitespace-only numeric cell',
                   'equality': 'Complete canonical bytes; SHA-256 for report labels'},
        'counts': {'shared_feature_keys': len(matches),
                   'overlapping_test_rows': len(feature_flagged),
                   'shared_groups': None if group_matches is None else len(group_matches),
                   'test_rows_in_shared_groups': None if group_matches is None else len(group_flagged),
                   'all_features_missing': missing_counts,
                   'digest_collision_buckets': collisions},
        'feature_matches': matches, 'group_matches': group_matches,
        'test_row_flags': [{'data_row': i, 'selected_features_seen_in_train': i in feature_flagged,
                            'group_seen_in_train': None if group_column is None else i in group_flagged}
                           for i in range(1, len(test) + 1)],
        'warnings': warnings,
        'limits': ['Potential overlap warning, not proof of leakage or a general split safety certificate.',
                   'No near-duplicate, chronology, target-proxy or unrecorded entity detection.',
                   'Row positions are one-based data records, not physical CSV line numbers.',
                   'Fingerprints are not anonymization; keep private-data reports private.'],
    }


def read_csv_snapshot(path: Path) -> tuple[list[dict], dict]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Input must be a regular CSV file of at most 5 MB.')
    try:
        with path.open('rb') as source:
            raw = source.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError('CSV exceeds the 5 MB input limit.')
        with io.StringIO(raw.decode('utf-8-sig'), newline='') as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames
            if not fields or len(fields) > MAX_COLUMNS or any(not f or len(f) > 128 for f in fields):
                raise ValueError('Invalid or missing CSV header.')
            if len(set(fields)) != len(fields):
                raise ValueError('Duplicate CSV header names.')
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('CSV record width differs from its header.')
                rows.append(row)
                if len(rows) > MAX_ROWS:
                    raise ValueError('CSV has more than 50000 data rows.')
    except (csv.Error, UnicodeError) as exc:
        raise ValueError('Input must be well-formed UTF-8 CSV.') from exc
    if not rows:
        raise ValueError('CSV needs at least one data row.')
    return rows, {'sha256': digest(raw), 'bytes': len(raw)}


def read_csv(path: Path) -> list[dict]:
    return read_csv_snapshot(path)[0]


def flags_csv(report: dict) -> str:
    stream = io.StringIO(newline='')
    columns = ['data_row', 'selected_features_seen_in_train', 'group_seen_in_train']
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(report['test_row_flags'])
    return stream.getvalue()


def write_report(output: Path, report: dict, *, input_paths=(), overwrite=False) -> None:
    targets = [output/'report.json', output/'test-row-flags.csv']
    protected = {Path(p).resolve() for p in input_paths}
    for p in targets:
        if p.resolve() in protected or p.is_symlink():
            raise ValueError('Report target conflicts with an input or symlink.')
        if p.exists() and not overwrite:
            raise ValueError('Report already exists; choose a new output directory or --overwrite.')
    body = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    table = flags_csv(report)
    output.mkdir(parents=True, exist_ok=True)
    targets[0].write_bytes(body.encode())
    targets[1].write_bytes(table.encode())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--numeric', nargs='*', default=[])
    parser.add_argument('--group')
    parser.add_argument('--target')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fail-on-overlap', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        train, train_metadata = read_csv_snapshot(args.train)
        test, test_metadata = read_csv_snapshot(args.test)
        report = audit(train, test, args.features,
                       numeric_columns=args.numeric, group_column=args.group,
                       target_column=args.target)
        report['input_files'] = {'train': train_metadata, 'test': test_metadata}
        write_report(args.output, report, input_paths=(args.train, args.test), overwrite=args.overwrite)
    except (ValueError, OSError, RecursionError) as exc:
        print(json.dumps({'status': 'input_error', 'message': str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'counts': report['counts']}, ensure_ascii=True))
    return 1 if args.fail_on_overlap and report['status'] == 'overlap_detected' else 0


if __name__ == '__main__':
    raise SystemExit(main())
