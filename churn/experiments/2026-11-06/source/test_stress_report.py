import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from churn_lab import FEATURES
from stress_report import (read_input,score_array,csv_text,write_outputs,metrics,validate_bundle)


def as_csv(rows,fields=None):
    fields=fields or ['record_id']+FEATURES
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n');writer.writerow(fields);writer.writerows(rows)
    return stream.getvalue()


class StressChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x=np.array([[i%2,0,10,1,100+i,2,3,4,2,1,25+i%5] for i in range(100)],dtype=float)
        y=np.array([i%2 for i in range(100)])
        with threadpool_limits(limits=1):
            model=HistGradientBoostingClassifier(max_iter=10,min_samples_leaf=5,random_state=13).fit(x,y)
        cls.x=x;cls.bundle={'schema':'churn-stress-v1','features':FEATURES,'model':model,'seed':13,
            'minimum':x.min(axis=0),'maximum':x.max(axis=0),'median':np.median(x,axis=0),'prior':float(y.mean())}

    def test_native_matches_direct_estimator_and_keeps_input(self):
        x=self.x[:5].copy();before=x.copy();r=score_array(x,list('abcde'),self.bundle,'native')
        with threadpool_limits(limits=1): expected=self.bundle['model'].predict_proba(x)[:,1]
        np.testing.assert_array_equal([a['model_probability'] for a in r['rows']],expected)
        np.testing.assert_array_equal(x,before)

    def test_missing_is_not_silently_zero_and_prior_is_separate(self):
        x=self.x[:2].copy();x[0,4]=np.nan
        r=score_array(x,['a','b'],self.bundle,'guarded')
        self.assertEqual((r['accepted'],r['rejected']), (1,1))
        self.assertIsNone(r['rows'][0]['model_probability'])
        self.assertEqual(r['rows'][0]['fallback_probability'],.5)
        self.assertIn('missing:Seconds of Use',json.loads(r['rows'][0]['input_flags']))
        self.assertIsNone(r['rows'][1]['fallback_probability'])

    def test_training_median_fills_only_missing(self):
        x=self.x[:2].copy();x[0,4]=np.nan
        expected=x.copy();expected[0,4]=self.bundle['median'][4]
        r=score_array(x,['a','b'],self.bundle,'median')
        with threadpool_limits(limits=1): probs=self.bundle['model'].predict_proba(expected)[:,1]
        np.testing.assert_array_equal([a['model_probability'] for a in r['rows']],probs)
        self.assertTrue(np.isnan(x[0,4]))

    def test_extrema_boundaries_and_shift_rejection(self):
        x=self.x[:2].copy();x[0]=self.bundle['minimum'];x[1,4]=self.bundle['maximum'][4]+1
        r=score_array(x,['a','b'],self.bundle,'guarded')
        self.assertEqual(r['accepted'],1)
        self.assertEqual(r['rows'][0]['input_flags'],'[]')
        self.assertIn('outside_fit_range:Seconds of Use',r['rows'][1]['input_flags'])

    def test_all_rejected_is_explicit_not_empty_model_prediction(self):
        r=score_array(np.full((2,11),np.nan),['b','a'],self.bundle,'guarded')
        self.assertEqual(r['coverage'],0)
        self.assertTrue(all(a['model_probability'] is None for a in r['rows']))
        self.assertEqual([a['record_id'] for a in r['rows'] if a['exact_10_percent_review']],['a'])

    def test_range_check_is_not_anomaly_certificate(self):
        x=self.x[:1].copy();x[0,4]=self.bundle['median'][4]
        r=score_array(x,['a'],self.bundle,'guarded')
        self.assertEqual(r['accepted'],1)
        self.assertIn('not reliability certification',r['interpretation'])

    def test_labels_never_change_inputs_or_scores(self):
        row=['a']+self.x[0].tolist()
        a,ids,ya=read_input(as_csv([row+[0]],['record_id']+FEATURES+['Churn']))
        b,_,yb=read_input(as_csv([row+[1]],['record_id']+FEATURES+['Churn']))
        np.testing.assert_array_equal(a,b);self.assertNotEqual(ya.tolist(),yb.tolist())
        self.assertEqual(score_array(a,ids,self.bundle,'guarded'),score_array(b,ids,self.bundle,'guarded'))

    def test_parser_blank_unknown_duplicate_missing_and_ragged(self):
        row=['a']+self.x[0].tolist();row[5]=''
        values,_,_=read_input(as_csv([row]));self.assertTrue(np.isnan(values[0,4]))
        invalid=[as_csv([row],['record_id']+FEATURES[:-1]+['unknown']),
            as_csv([row],['record_id']+FEATURES[:-1]+[FEATURES[0]]),
            as_csv([row[:-1]]),as_csv([row+['extra']]),as_csv([]),as_csv([row,row])]
        for text in invalid:
            with self.assertRaises(ValueError):read_input(text)

    def test_parser_rejects_nonfinite_nonnumeric_and_invalid_ids(self):
        row=['a']+self.x[0].tolist()
        for value in ('nan','inf','-inf','text'):
            wrong=row.copy();wrong[1]=value
            with self.assertRaises(ValueError):read_input(as_csv([wrong]))
        row[0]='=formula'
        with self.assertRaises(ValueError):read_input(as_csv([row]))
        with self.assertRaises(ValueError):read_input('x'*1_000_001)

    def test_metrics_empty_single_class_ties_and_capacity(self):
        m=metrics(np.array([1,0,1,0]),np.array([.8,.8,.2,.1]),['z','a','c','d'])
        self.assertAlmostEqual(m['ap'],(1/2+2/3)/2)
        self.assertAlmostEqual(m['brier'],(.04+.64+.64+.01)/4)
        self.assertEqual(m['capacity']['tp'],0)
        self.assertIsNone(metrics(np.array([]),np.array([]),[])['ap'])
        self.assertIsNone(metrics(np.array([1]),np.array([.1]),['a'])['ap'])

    def test_export_preserves_null_boolean_flags_and_probabilities(self):
        x=self.x[:2].copy();x[0,4]=np.nan;r=score_array(x,['a','b'],self.bundle,'guarded')
        rows=list(csv.DictReader(csv_text(r).splitlines()))
        for actual,expected in zip(rows,r['rows']):
            for k in ['model_probability','fallback_probability','effective_policy_score']:
                self.assertEqual(None if actual[k]=='' else float(actual[k]),expected[k])
            self.assertEqual(actual['input_flags'],expected['input_flags'])
            self.assertEqual(actual['exact_10_percent_review']=='True',expected['exact_10_percent_review'])

    def test_output_preserves_existing_input_and_bundle(self):
        r=score_array(self.x[:1],['a'],self.bundle,'guarded')
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);protected=p/'input.csv';protected.write_text('unique')
            with self.assertRaises(ValueError):write_outputs(r,protected,p/'out.csv',[protected])
            with self.assertRaises(ValueError):write_outputs(r,p/'same',p/'same')
            write_outputs(r,p/'out.json',p/'out.csv')
            with self.assertRaises(FileExistsError):write_outputs(r,p/'out.json',p/'other.csv')
            self.assertFalse((p/'other.csv').exists());self.assertEqual(protected.read_text(),'unique')

    def test_invalid_bundle_policy_shapes_and_infinities(self):
        for bundle in [{**self.bundle,'features':list(reversed(FEATURES))},
            {**self.bundle,'median':[np.nan]*11},{**self.bundle,'prior':2}]:
            with self.assertRaises(ValueError):validate_bundle(bundle)
        with self.assertRaises(ValueError):score_array(self.x[:1],['a'],self.bundle,'wrong')
        with self.assertRaises(ValueError):score_array(self.x[:1],['a','b'],self.bundle,'native')
        with self.assertRaises(ValueError):score_array(np.full((1,11),np.inf),['a'],self.bundle,'native')


if __name__=='__main__':unittest.main(verbosity=2)
