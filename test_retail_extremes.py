"""Authored correctness fixtures; these do not replace the real transaction study."""
import json
import math
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from retail_extremes import (METHODS, fit_pipeline, predict, prepare, stress_spend,
                             partition_comparison, quantile_scalar)
from retail_rfm import clean_transactions, rfm, scalar_rfm_check, profiles


class ExtremeChecks(unittest.TestCase):
    def test_label_permutations_and_exact_ari(self):
        c=partition_comparison([0,0,1,1],[1,1,0,0])
        self.assertEqual(c['ari'],1.)
        self.assertEqual(c['optimal_label_aligned_agreement'],1.)
        self.assertEqual(partition_comparison([0,0,1,1],[0,1,0,1])['ari'],-.5)

    def test_ties_and_mutation_locality(self):
        f=pd.DataFrame([[1,1,10],[2,2,10],[3,3,1]],columns=['R','F','M'],index=['b','a','c']);before=f.copy(deep=True)
        out,positions=stress_spend(f,fraction=1/3,multiplier=10)
        self.assertEqual(positions.tolist(),[1]);self.assertEqual(out.loc['a','M'],100)
        pd.testing.assert_frame_equal(f,before)
        np.testing.assert_array_equal(out.iloc[:,:2],f.iloc[:,:2])
        self.assertEqual(out.loc['b','M'],10)

    def test_zero_full_fraction_and_bad_stress(self):
        f=pd.DataFrame([[1,1,1],[2,2,2]],columns=['R','F','M'])
        a,i=stress_spend(f,0);pd.testing.assert_frame_equal(a,f);self.assertEqual(len(i),0)
        a,i=stress_spend(f,1);self.assertEqual(len(i),2);np.testing.assert_array_equal(a.M,f.M*10)
        for fraction,multiplier in [(-.1,10),(1.1,10),(.1,0),(.1,float('inf'))]:
            with self.assertRaises(ValueError):stress_spend(f,fraction,multiplier)

    def test_four_pipelines_and_reloaded_parameters(self):
        x=np.array([[1,1,2],[2,1,3],[20,10,300],[21,11,400]],dtype=float)
        with threadpool_limits(limits=1):
            for method in METHODS:
                bundle,record=fit_pipeline(x,method)
                self.assertTrue(record['valid_two_groups'])
                np.testing.assert_array_equal(predict(bundle,x),predict(json.loads(json.dumps(bundle)),x))

    def test_scaler_fits_training_rows_only(self):
        x=np.array([[1,1,2],[2,2,4],[3,3,6],[4,4,8]],dtype=float)
        with threadpool_limits(limits=1):bundle,_=fit_pipeline(x,'raw_robust')
        saved=json.dumps(bundle,sort_keys=True)
        predict(bundle,[[1,1,1e9],[2,2,3e9]])
        self.assertEqual(saved,json.dumps(bundle,sort_keys=True))
        self.assertEqual(bundle['center'],[2.5,2.5,5.])
        self.assertEqual(bundle['scale'],[1.5,1.5,3.])

    def test_invalid_feature_arrays(self):
        for values in [[[1,2,3]],[[1,2],[3,4]],[[1,-1,2],[2,3,4]],[[1,2,float('nan')],[2,3,4]]]:
            with self.assertRaises(ValueError):prepare(values,'log_standard')
        with self.assertRaises(ValueError):prepare([[1,2,3],[4,5,6]],'unknown')

    def test_independent_quantile_interpolation(self):
        self.assertEqual(quantile_scalar([100,1,2,3],.25),1.75)
        self.assertEqual(quantile_scalar([100,1,2,3],.5),2.5)

    def test_cleaning_dates_duplicates_and_rfm_oracle(self):
        rows=[['1','a','2011-01-01',2,3],['1','a','2011-01-01',2,3],
              ['1','C2','2011-01-02',2,3],['1','b','2011-01-03',-1,3],
              [None,'c','2011-01-02',1,2],['1','d','2011-02-01',1,100]]
        frame=pd.DataFrame(rows,columns=['CustomerID','InvoiceNo','InvoiceDate','Quantity','UnitPrice'])
        clean,counts=clean_transactions(frame);f,chosen=rfm(clean,'2011-01-01','2011-02-01')
        self.assertEqual(counts['retained_rows'],3);self.assertEqual(f.loc['1','F'],1);self.assertEqual(f.loc['1','M'],12)
        self.assertEqual(scalar_rfm_check(f,chosen,'2011-02-01'),0.)

    def test_small_profiles_suppressed(self):
        f=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns=['R','F','M'])
        records=profiles(f,np.array([0,0,1]))
        self.assertTrue(all(r['small_profile_suppressed'] and r['median_rfm'] is None for r in records))


if __name__=='__main__':unittest.main()
