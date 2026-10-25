import unittest
import numpy as np
from imbalance_experiment import oversample_indices


class SamplingChecks(unittest.TestCase):
    def test_originals_preserved_and_only_minority_added(self):
        y=np.array([0,0,0,0,1]);idx,w=oversample_indices(y,13)
        self.assertEqual(idx.tolist(),[0,1,2,3,4,4,4,4]);self.assertAlmostEqual(w.sum(),5)
        np.testing.assert_equal(np.bincount(y[idx]),[4,4])
    def test_seed_and_no_input_mutation(self):
        y=np.array([0,0,0,0,0,1,1]);copy=y.copy();a,_=oversample_indices(y,29);b,_=oversample_indices(y,29)
        np.testing.assert_equal(a,b);np.testing.assert_equal(y,copy)
    def test_balanced_no_duplication(self):
        idx,w=oversample_indices(np.array([1,0]),13);np.testing.assert_equal(idx,[0,1]);np.testing.assert_equal(w,[1,1])
    def test_invalid_labels(self):
        for y in [[],[0,0],[1,2],[[0,1]]]:
            with self.assertRaises((ValueError,TypeError)):oversample_indices(y,13)


if __name__=='__main__':unittest.main()
