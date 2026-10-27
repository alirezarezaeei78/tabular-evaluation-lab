import tempfile
import unittest
import numpy as np
from sklearn.cluster import DBSCAN
from example import inputs,fit,validate,density_oracle,run


class ClusteringTeachingChecks(unittest.TestCase):
    def test_repeatable_seed_and_distinct_probes(self):
        np.testing.assert_array_equal(inputs(),inputs())
        self.assertEqual(inputs().shape,(244,2));self.assertEqual(inputs(40,29).shape,(84,2))
    def test_exact_self_count_neighborhood_boundary(self):
        x=np.array([[0.,0.],[.1,0.]])
        for minimum,noise in [(2,0),(3,2)]:
            model=DBSCAN(eps=.11,min_samples=minimum).fit(x)
            summary=density_oracle(x,model.labels_,model.core_sample_indices_,eps=.11,min_samples=minimum)
            self.assertEqual(summary['noise_rows'],noise)
    def test_independent_components_border_and_noise(self):
        x=np.array([[0,0],[.05,0],[0,.05],[.05,.05],[.15,0],[3,3]],dtype=float)
        model=DBSCAN(eps=.11,min_samples=4).fit(x)
        summary=density_oracle(x,model.labels_,model.core_sample_indices_,eps=.11,min_samples=4)
        self.assertEqual(summary['core_components'],1);self.assertEqual(summary['noise_rows'],1)
    def test_invalid_shapes_and_nonfinite_rows(self):
        for x in [[],[1,2],[[0,0]],[[0,0,0],[1,1,1]],[[0,0],[1,float('nan')]]]:
            with self.assertRaises(ValueError):validate(x)
    def test_duplicate_rows_are_valid_and_counted(self):
        x=np.array([[0,0],[0,0],[1,1],[1,1]],dtype=float)
        model=DBSCAN(eps=.01,min_samples=2).fit(x)
        summary=density_oracle(x,model.labels_,model.core_sample_indices_,eps=.01,min_samples=2)
        self.assertEqual(summary['core_components'],2)
    def test_fit_and_export_roundtrip_and_oracles(self):
        with tempfile.TemporaryDirectory() as d:r=run(d)
        self.assertTrue(r['checks']['held_out_transform_does_not_change_fit'])
        self.assertTrue(r['density']['independent_graph_checks'])
        self.assertFalse(r['checks']['dbscan_has_builtin_predict'])
    def test_same_seed_same_partitions(self):
        a=fit(inputs());b=fit(inputs())
        np.testing.assert_array_equal(a[2].labels_,b[2].labels_)
        np.testing.assert_array_equal(a[3].labels_,b[3].labels_)


if __name__=='__main__':unittest.main(verbosity=2)
