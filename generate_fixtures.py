"""Create declared synthetic CSVs and an oracle independent of auditor hashes."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def dump(path, obj):
    path.write_bytes((json.dumps(obj, indent=2) + '\n').encode())


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['row_id', 'entity', 'measurement', 'channel', 'target'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())


def expected_counts(train_indices, test_indices, metadata):
    # Latent identities from construction, not numeric parsing, canonicalization or hashes.
    training_features = {metadata[i]['feature_key'] for i in train_indices}
    testing_features = {metadata[i]['feature_key'] for i in test_indices}
    training_groups = {metadata[i]['entity_key'] for i in train_indices}
    testing_groups = {metadata[i]['entity_key'] for i in test_indices}
    return {'shared_feature_keys': len(training_features & testing_features),
            'overlapping_test_rows': sum(metadata[i]['feature_key'] in training_features for i in test_indices),
            'shared_groups': len(training_groups & testing_groups),
            'test_rows_in_shared_groups': sum(metadata[i]['entity_key'] in training_groups for i in test_indices)}


def generate(output=ROOT/'fixtures'):
    config = json.loads((ROOT/'config.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    if (output/'fixture-manifest.json').exists():
        raise ValueError('Fixtures already declared; use saved files instead of overwriting.')
    generator = random.Random(config['generator_seed'])
    rows, metadata = [], []
    for group in range(config['entities']):
        label = str(generator.randrange(2))
        for visit in range(3):
            distinct = visit == 2
            # Two equivalent numeric encodings are intentionally mixed.
            value = str(group*10)+('.25' if distinct else ('.00' if visit == 1 else ''))
            rows.append({'row_id': f'r{group:03d}-{visit}', 'entity': f'e{group:03d}',
                         'measurement': value, 'channel': f'channel-{group % 4}', 'target': label})
            metadata.append({'entity_key': group, 'feature_key': group*2+int(distinct)})
    cases = []
    for seed in config['split_seeds']:
        for method in ['random_rows', 'feature_partition', 'group_partition']:
            rng = random.Random(seed)
            if method == 'random_rows':
                chosen = set(rng.sample(range(len(rows)), int(len(rows)*config['test_fraction'])))
            else:
                field = 'feature_key' if method == 'feature_partition' else 'entity_key'
                keys = sorted({m[field] for m in metadata})
                chosen_keys = set(rng.sample(keys, int(len(keys)*config['test_fraction'])))
                chosen = {i for i, m in enumerate(metadata) if m[field] in chosen_keys}
            train_indices = [i for i in range(len(rows)) if i not in chosen]
            test_indices = [i for i in range(len(rows)) if i in chosen]
            assert set(train_indices).isdisjoint(test_indices)
            assert sorted(train_indices+test_indices) == list(range(600))
            case_id = f'{method}-seed-{seed}'
            folder = output/case_id
            folder.mkdir()
            write_csv(folder/'train.csv', [rows[i] for i in train_indices])
            write_csv(folder/'test.csv', [rows[i] for i in test_indices])
            cases.append({'id': case_id, 'method': method, 'seed': seed,
                          'train_file': f'{case_id}/train.csv', 'test_file': f'{case_id}/test.csv',
                          'expected': expected_counts(train_indices, test_indices, metadata),
                          'train_indices': train_indices, 'test_indices': test_indices,
                          'all_rows_retained': True})
    cross_train = [{'row_id': 'a1', 'entity': 'a', 'measurement': '1', 'channel': 'common', 'target': '0'},
                   {'row_id': 'b1', 'entity': 'b', 'measurement': '2', 'channel': 'other', 'target': '1'}]
    cross_test = [{'row_id': 'c1', 'entity': 'c', 'measurement': '1.00', 'channel': 'common', 'target': '1'},
                  {'row_id': 'd1', 'entity': 'd', 'measurement': '3', 'channel': 'third', 'target': '0'}]
    folder = output/'independent_common_values'
    folder.mkdir()
    write_csv(folder/'train.csv', cross_train)
    write_csv(folder/'test.csv', cross_test)
    cases.append({'id': 'independent_common_values', 'method': 'independent_common_values', 'seed': None,
                  'train_file': 'independent_common_values/train.csv', 'test_file': 'independent_common_values/test.csv',
                  'expected': {'shared_feature_keys': 1, 'overlapping_test_rows': 1, 'shared_groups': 0,
                               'test_rows_in_shared_groups': 0},
                  'all_rows_retained': True,
                  'interpretation': 'Independent entity IDs share common selected features; a warning is not proof of contamination.'})
    dump(output/'cases.json', cases)
    dump(output/'latent-identities.json', metadata)
    dump(output/'fixture-manifest.json', {'generated_from': config, 'licence': 'MIT; freshly authored synthetic data',
         'files': {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(output.rglob('*')) if p.is_file()}})
    return cases


if __name__ == '__main__':
    print(json.dumps({'generated_cases': len(generate())}))
