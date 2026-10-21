"""Meaningful numerical, policy, leakage and local HTTP workflow checks.

Synthetic classifier fixtures verify implementation, never benchmark accuracy.
"""
import csv
import io
import json
import math
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from http.server import ThreadingHTTPServer

from churn_lab import (FEATURES, MODEL_NAMES, checked_probabilities, confusion,
    development_cutoff, evaluate, exact_capacity, export_csv, oracle_ap, parse_csv,
    reliability, score_csv, split_indices)
from server import make_handler

HERE=Path(__file__).resolve().parent


def fixture_bundle():
    rng=np.random.default_rng(7);x=rng.normal(size=(200,len(FEATURES)));y=(x[:,0]+x[:,1]>.2).astype(int)
    models={'dummy':DummyClassifier(strategy='prior'),
            'logistic':make_pipeline(StandardScaler(),LogisticRegression(random_state=7)),
            'boosting':HistGradientBoostingClassifier(max_iter=10,early_stopping=False,random_state=7)}
    with threadpool_limits(limits=1):
        for m in models.values():m.fit(x[:120],y[:120])
        models['boosting_sigmoid']=CalibratedClassifierCV(models['boosting'],method='sigmoid',cv='prefit').fit(x[120:170],y[120:170])
    return {'models':models,'cutoffs':{k:development_cutoff(v.predict_proba(x[120:170])[:,1]) for k,v in models.items()}}


class NumericalAndPolicy(unittest.TestCase):
    def test_tie_aware_ap_against_sklearn(self):
        y=np.array([1,0,1,0,1,0]);p=np.array([.8,.8,.8,.2,.1,.1])
        self.assertAlmostEqual(oracle_ap(y,p),average_precision_score(y,p),places=14)

    def test_same_score_ap_is_prevalence(self):
        self.assertAlmostEqual(oracle_ap([1,0,0,0],[.5]*4),.25)

    def test_brier_and_counts_have_hand_calculated_answer(self):
        d=evaluate(np.array([1,0]),np.array([.8,.3]),['a','b'],.6)
        self.assertAlmostEqual(d['brier'],.065)
        self.assertEqual((d['policies']['cutoff_0.5']['tp'],d['policies']['cutoff_0.5']['fp']),(1,0))

    def test_confusion_oracle_all_four_cells(self):
        x=confusion([1,1,0,0],[True,False,True,False])
        self.assertEqual([x[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])

    def test_undefined_precision_is_explicit(self):
        d=confusion([1,0],[False,False]);self.assertIsNone(d['precision']);self.assertEqual(d['recall'],0)

    def test_decimal_capacity_does_not_over_select(self):
        self.assertEqual(exact_capacity(np.arange(100)/100,[str(i) for i in range(100)],.07).sum(),7)

    def test_capacity_small_batch_rounds_up(self):
        self.assertEqual(exact_capacity([.3,.5,.1],['a','b','c'],.1).tolist(),[False,True,False])

    def test_ties_are_id_deterministic(self):
        self.assertEqual(exact_capacity([.4,.4,.4],['z','a','m'],1/3).tolist(),[False,True,False])

    def test_capacity_zero_and_full(self):
        self.assertEqual(exact_capacity([.3,.9],['a','b'],0).sum(),0)
        self.assertEqual(exact_capacity([.3,.9],['a','b'],1).sum(),2)

    def test_empty_capacity_is_valid_zero(self):
        self.assertEqual(len(exact_capacity([],[],.1)),0)

    def test_probability_and_budget_failure_cases(self):
        for p in [[math.nan],[math.inf],[-.1],[1.1],[[.5]]]:
            with self.subTest(p=p),self.assertRaises(ValueError):checked_probabilities(p)
        for f in [-.1,1.1,math.nan]:
            with self.subTest(f=f),self.assertRaises(ValueError):exact_capacity([.3],['a'],f)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):exact_capacity([.1,.2],['a','a'],.1)

    def test_development_threshold_ties_not_exact_capacity(self):
        threshold=development_cutoff([.9,.9,.2,.1],.25)
        self.assertEqual(threshold,.9)
        self.assertEqual(sum(np.array([.9,.9,.2,.1])>=threshold),2)
        self.assertEqual(exact_capacity([.9,.9,.2,.1],['a','b','c','d'],.25).sum(),1)

    def test_reliability_bins_include_probability_one(self):
        rows=reliability(np.array([0,1]),np.array([0.,1.]));self.assertEqual(sum(r['n'] for r in rows),2)
        self.assertEqual(rows[-1]['mean_score'],1)

    def test_group_split_keeps_conflicting_duplicate_labels_together(self):
        groups=np.repeat(np.arange(200),2);x=groups[:,None];y=np.tile([0,1],200)
        p=split_indices(x,y,groups,13)
        self.assertEqual(sum(len(v) for v in p.values()),400)
        where={int(i):name for name,idx in p.items() for i in idx}
        self.assertTrue(all(where[i]==where[i+1] for i in range(0,400,2)))

    def test_selection_does_not_use_outcomes(self):
        p=[.9,.5,.2];before=exact_capacity(p,['a','b','c'],.34)
        # Outcomes can reverse without changing the score-only review policy.
        self.assertNotEqual(confusion([1,0,0],before)['tp'],confusion([0,0,1],before)['tp'])
        np.testing.assert_array_equal(before,exact_capacity(p,['a','b','c'],.34))


class LocalWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=fixture_bundle();cls.demo=(HERE/'examples/demo.csv').read_text(encoding='utf-8')
        cls.temp=tempfile.TemporaryDirectory(); report=Path(cls.temp.name)/'report.json';report.write_text('{"test_fixture":true}')
        cls.httpd=ThreadingHTTPServer(('127.0.0.1',0),make_handler(cls.bundle,report))
        cls.thread=threading.Thread(target=cls.httpd.serve_forever,daemon=True);cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.httpd.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown();cls.httpd.server_close();cls.thread.join();cls.temp.cleanup()

    def request(self,payload,**headers):
        req=urllib.request.Request(self.base+'/api/score',json.dumps(payload).encode(),headers={'Content-Type':'application/json',**headers})
        with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)

    def test_authored_demo_schema_and_all_models(self):
        self.assertEqual(len(parse_csv(self.demo)),10)
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                d=score_csv(self.demo,self.bundle,name,.1)
                self.assertEqual(d['n'],10);self.assertEqual(d['exact_capacity_count'],1)

    def test_model_receives_only_eleven_features(self):
        self.assertEqual(len(FEATURES),11)
        self.assertFalse(set(FEATURES)&{'Churn','record_id','Status','Customer Value'})

    def test_unknown_target_proxy_and_missing_columns_rejected(self):
        for text in [self.demo.replace('Call Failure','Status'),self.demo.replace('Complains','Customer Value'),self.demo.replace('Age\n','Churn\n',1)]:
            with self.subTest(text=text[:60]),self.assertRaises(ValueError):parse_csv(text)

    def test_nan_inf_and_fractional_features_rejected(self):
        for value in ['NaN','Inf','2.5','-1']:
            with self.subTest(value=value),self.assertRaises(ValueError):parse_csv(self.demo.replace('demo-a,2,','demo-a,'+value+',',1))

    def test_duplicate_ids_formula_ids_and_empty_data_rejected(self):
        for text in [self.demo.replace('demo-b','demo-a'),self.demo.replace('demo-a','=SUM'),self.demo.splitlines()[0]+'\n']:
            with self.subTest(text=text[-30:]),self.assertRaises(ValueError):parse_csv(text)

    def test_observed_charge_amount_ten_is_accepted(self):
        text=self.demo.replace('demo-a,2,0,18,2,','demo-a,2,0,18,10,',1)
        self.assertEqual(parse_csv(text)[0]['Charge Amount'],10)
        with self.assertRaises(ValueError):parse_csv(text.replace('demo-a,2,0,18,10,','demo-a,2,0,18,11,',1))

    def test_csv_json_export_full_precision_and_flags(self):
        data=score_csv(self.demo,self.bundle)
        rows=list(csv.DictReader(io.StringIO(export_csv(data))))
        for a,b in zip(rows,data['rows']):
            self.assertEqual(float(a['score']),b['score']);self.assertEqual(a['exact_capacity_review'],str(b['exact_capacity_review']))
        self.assertEqual(json.loads(json.dumps(data)),data)

    def test_real_http_workflow_and_export_parity(self):
        d=self.request({'csv':self.demo,'capacity':.2,'model':'logistic'})
        self.assertEqual(d['report']['exact_capacity_count'],2)
        self.assertEqual(d['csv'],export_csv(d['report']))

    def test_http_invalid_input_has_explicit_error(self):
        with self.assertRaises(urllib.error.HTTPError) as c:self.request({'csv':'bad'})
        self.assertEqual(c.exception.code,400)
        self.assertIn('error',json.load(c.exception))

    def test_cross_origin_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as c:self.request({'csv':self.demo},Origin='https://external.invalid')
        self.assertEqual(c.exception.code,403)

    def test_source_target_changes_cannot_change_scores(self):
        lines=self.demo.splitlines();a=lines[0]+',Churn\n'+'\n'.join(x+',0' for x in lines[1:])
        b=lines[0]+',Churn\n'+'\n'.join(x+',1' for x in lines[1:])
        da=score_csv(a,self.bundle);db=score_csv(b,self.bundle)
        self.assertEqual([x['score'] for x in da['rows']],[x['score'] for x in db['rows']])
        self.assertEqual([x['exact_capacity_review'] for x in da['rows']],[x['exact_capacity_review'] for x in db['rows']])


if __name__=='__main__':unittest.main(verbosity=2)
