import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from segment_snapshot import compare, export, snapshot, model


class SnapshotChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Synthetic hand-set centers: origin vs log(10) in each dimension.
        self.obj = dict(schema='frozen-rfm-v1', method='log_standard', center=[0]*3,
                        scale=[1]*3, centers=[[0]*3, [np.log(10)]*3], development_medians=[5,5,5])
        self.m = self.root/'model.json'
        self.m.write_text(json.dumps(self.obj))
        self.a = self.write('a.csv', [['a',1,1,1], ['b',9,9,9], ['c',1,1,1], ['left',9,9,9]])
        self.b = self.write('b.csv', [['c',9,9,9], ['b',9,9,9], ['a',1,1,1], ['right',1,1,1]])

    def write(self, name, rows):
        p=self.root/name
        with p.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['CustomerID','R','F','M']);w.writerows(rows)
        return p

    def test_hand_calculated_join_and_metrics(self):
        r,rows=compare(self.a,self.b,self.m)
        self.assertEqual(r['accounting']['matched'],3)
        self.assertEqual(r['accounting']['union'],5)
        self.assertEqual(r['accounting']['left_only'],1)
        self.assertEqual(r['accounting']['right_only'],1)
        for method in ['kmeans','median_rule']:
            m=r['methods'][method]
            self.assertEqual(m['changed_fixed_labels'],1)
            self.assertEqual(m['changed_fixed_label_share'],1/3)
            self.assertAlmostEqual(m['ari'],-.5)
            self.assertEqual([x['count'] for x in m['transitions']],[1,1,0,1])
        self.assertEqual([x['CustomerID'] for x in rows],['a','b','c','left','right'])

    def test_reordering_and_no_input_mutation(self):
        original=self.a.read_bytes(),self.b.read_bytes(),self.m.read_bytes()
        r,rows=compare(self.a,self.b,self.m)
        reverse=self.write('reverse.csv',[['c',1,1,1],['left',9,9,9],['b',9,9,9],['a',1,1,1]])
        s,other=compare(reverse,self.b,self.m)
        r.pop('input_sha256');s.pop('input_sha256')
        self.assertEqual(r,s);self.assertEqual(rows,other)
        self.assertEqual(original,(self.a.read_bytes(),self.b.read_bytes(),self.m.read_bytes()))

    def test_empty_disjoint_and_single(self):
        for rows,n in [([],0),([['other',1,1,1]],0),([['a',1,1,1]],1)]:
            p=self.write('edge.csv',rows);r,_=compare(self.a,p,self.m)
            self.assertEqual(r['accounting']['matched'],n)
            self.assertIsNone(r['methods']['kmeans']['ari'])
            self.assertEqual(r['methods']['kmeans']['changed_fixed_label_share'],0 if n else None)

    def test_invalid_rows(self):
        cases=[[['a',1,1,1],['a',2,2,2]],[[' a',1,1,1]],[['',1,1,1]],
               [['a','nan',1,1]],[['a',1,1.2,1]],[['a',-1,1,1]],[['a',1,1,0]]]
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):snapshot(self.write('bad.csv',rows))

    def test_invalid_model(self):
        for field,value in [('scale',[1,0,1]),('centers',[[0,0]]),('center',[0,float('nan'),0]),('method','raw_standard')]:
            obj={**self.obj,field:value};self.m.write_text(json.dumps(obj))
            with self.subTest(field=field),self.assertRaises(ValueError):model(self.m)

    def test_exports_match_report(self):
        report,rows=compare(self.a,self.b,self.m);out=self.root/'out';export(report,rows,out)
        self.assertEqual(json.loads((out/'report.json').read_text()),report)
        with (out/'assignments.csv').open() as f: exported=list(csv.DictReader(f))
        self.assertEqual(len(exported),5)
        for row in exported:
            expected=next(x for x in rows if x['CustomerID']==row['CustomerID'])
            self.assertEqual(row,{k:'' if v is None else str(v) for k,v in expected.items()})
        with (out/'transitions.csv').open() as f:
            for row in csv.DictReader(f):
                expected=next(x for x in report['methods'][row['method']]['transitions'] if x['left_label']==int(row['left_label']) and x['right_label']==int(row['right_label']))
                self.assertEqual(int(row['count']),expected['count'])
        with (out/'profiles.csv').open() as f:
            for row,expected in zip(csv.DictReader(f),report['profiles']):
                self.assertEqual(row,{k:'' if v is None else str(v) for k,v in expected.items()})
        with self.assertRaises(FileExistsError):export(report,rows,out)

    def test_non_object_model_and_bad_header(self):
        self.m.write_text('[]')
        with self.assertRaises(ValueError):model(self.m)
        self.a.write_text('CustomerID,R,F,M,M\na,1,1,1,2\n')
        with self.assertRaises(ValueError):snapshot(self.a)

    def test_profile_support_and_medians(self):
        a=self.write('large.csv',[[f'id{i}',1,1,1+i/100] for i in range(10)])
        report,_=compare(a,a,self.m)
        p=next(x for x in report['profiles'] if x['method']=='kmeans' and x['side']=='left' and x['scope']=='all' and x['label']==0)
        self.assertEqual(p['count'],10);self.assertFalse(p['medians_suppressed'])
        self.assertEqual(p['median_R'],1);self.assertEqual(p['median_F'],1)
        self.assertEqual(p['median_M'],1.045)
        self.assertTrue(all(x['medians_suppressed'] for x in report['profiles'] if x['count']<10))

    def test_cli_success_and_error_exit(self):
        cmd=[sys.executable,str(Path(__file__).with_name('segment_snapshot.py')),'--left',str(self.a),'--right',str(self.b),'--model',str(self.m),'--output',str(self.root/'cli')]
        first=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertEqual(json.loads(first.stdout)['accounting']['matched'],3)
        second=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(second.returncode,2);self.assertIn('error',second.stderr)


if __name__=='__main__':unittest.main()
