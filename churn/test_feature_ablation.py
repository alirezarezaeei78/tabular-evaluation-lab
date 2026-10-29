import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from churn_lab import FEATURES
from feature_ablation import GROUPS, choose_feature, equivalent_rows, export_slice, variants


class FeatureAblationChecks(unittest.TestCase):
    def test_known_correlation_and_constant(self):
        index, corr = choose_feature([[5, 0, 1], [5, 0, 1], [5, 1, 0], [5, 1, 0]], [0, 0, 1, 1])
        self.assertEqual(index, 1)
        np.testing.assert_allclose(corr, [0, 1, -1])

    def test_groups_partition_original_inputs(self):
        flattened = sum(GROUPS.values(), [])
        self.assertCountEqual(flattened, FEATURES)
        self.assertEqual(len(flattened), len(set(flattened)))
        for name, cols in variants(1).items():
            self.assertTrue(cols)
            self.assertNotIn('Churn', [FEATURES[i] for i in cols])

    def test_removed_input_cannot_change_fitted_predictions(self):
        rng = np.random.default_rng(8); x = rng.normal(size=(120, 11)); y = (x[:, 4] > 0).astype(int)
        cols = variants(1)['without_quality']
        model = HistGradientBoostingClassifier(max_iter=8, min_samples_leaf=5, early_stopping=False).fit(x[:, cols], y)
        before = model.predict_proba(x[:, cols]); x[:, :2] = 1e6
        np.testing.assert_array_equal(before, model.predict_proba(x[:, cols]))

    def test_collisions_count_rows_not_unique_tuples(self):
        self.assertEqual(equivalent_rows([[1, 2], [1, 2], [3, 4]], [[1, 2], [1, 2], [9, 9]]), 2)

    def test_actual_export_rows_match_and_reject_overwrite(self):
        report = {'summary_rows': [{'seed': 13, 'variant': 'full', 'ap': .25}, {'seed': 29, 'variant': 'full', 'ap': .5}]}
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); export_slice(report, 13, p / 'slice.csv'); export_slice(report, 13, p / 'slice.json')
            csvrow = list(csv.DictReader((p / 'slice.csv').read_text().splitlines()))[0]
            jsonrow = json.loads((p / 'slice.json').read_text())[0]
            self.assertEqual(float(csvrow['ap']), jsonrow['ap']); self.assertEqual(csvrow['variant'], jsonrow['variant'])
            with self.assertRaises(FileExistsError): export_slice(report, 13, p / 'slice.csv')
            with self.assertRaises(ValueError): export_slice(report, 47, p / 'absent.csv')
            with self.assertRaises(ValueError): export_slice(report, 13, p / 'wrong.txt')

    def test_invalid_fitting_inputs(self):
        for x, y in [([], []), ([[1], [2]], [0, 0]), ([[1], [float('nan')]], [0, 1])]:
            with self.assertRaises(ValueError): choose_feature(x, y)


if __name__ == '__main__': unittest.main()
