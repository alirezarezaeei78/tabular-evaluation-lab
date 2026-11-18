"""Reproduce the frozen snapshot app evaluation; provider artifacts stay local."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from retail_extremes import fit_pipeline, predict, INPUT_SHA
from retail_rfm import clean_transactions, rfm, scalar_rfm_check
from segment_snapshot import compare, export, digest


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(provider,old_models,development,reference_predictions,audit,public):
    audit.mkdir(parents=True,exist_ok=False);public.mkdir(parents=True,exist_ok=True)
    assert digest(provider.read_bytes())==INPUT_SHA
    frame=pd.read_excel(provider,engine='openpyxl');clean,cleaning=clean_transactions(frame)
    dev=pd.read_csv(development,index_col=0,float_precision='round_trip')
    recovered=json.loads(old_models.read_text(encoding='utf-8'))['log_standard_clean']
    bundle={**recovered,'schema':'frozen-rfm-v1','development_medians':dev.median().tolist(),
            'provenance':'Recovered clean 12 November log_standard model; rule medians from same development rows'}
    model_path=audit/'model.json';write_json(model_path,bundle)
    checks={};snapshots={}
    for days in [180,30]:
        features,rows=rfm(clean,pd.Timestamp('2011-12-01')-pd.Timedelta(days=days),'2011-12-01')
        checks['scalar_rfm_'+str(days)]=scalar_rfm_check(features,rows,'2011-12-01')
        path=audit/f'{days}-day.csv';features.to_csv(path,float_format='%.17g');snapshots[days]=(features,path)
    report,assignments=compare(snapshots[180][1],snapshots[30][1],model_path)
    export(report,assignments,audit/'api')
    cmd=[sys.executable,'segment_snapshot.py','--left',str(snapshots[180][1]),'--right',str(snapshots[30][1]),'--model',str(model_path),'--output',str(audit/'cli')]
    result=subprocess.run(cmd,capture_output=True,text=True);(audit/'cli-stdout.txt').write_text(result.stdout+result.stderr)
    assert result.returncode==0
    for name in ['report.json','assignments.csv','transitions.csv','profiles.csv']:
        assert (audit/'api'/name).read_bytes()==(audit/'cli'/name).read_bytes()
    prior=pd.read_csv(reference_predictions,dtype={'CustomerID':str})
    labels={x['CustomerID']:x['left_kmeans'] for x in assignments if x['left_kmeans'] is not None}
    assert all(labels[row.CustomerID]==row.log_standard_clean for row in prior.itertuples())
    # Independently use recovered pipeline function and direct median votes.
    for side,(features,path) in zip(['left','right'],snapshots.values()):
        expected=predict(recovered,features)
        actual={x['CustomerID']:x[side+'_kmeans'] for x in assignments}
        assert all(actual[str(key)]==int(label) for key,label in zip(features.index,expected))
        med=dev.median().to_numpy()
        for key,x in zip(features.index,features.to_numpy()):
            expected_rule=int(sum([x[0]<=med[0],x[1]>=med[1],x[2]>=med[2]])>=2)
            row=next(r for r in assignments if r['CustomerID']==str(key))
            assert row[side+'_median_rule']==expected_rule
    # Safe default fixtures: authored, unrelated to provider records.
    demo=public/'demo';demo.mkdir(exist_ok=True)
    train=np.array([[1+i%5,8+i%7,100+20*i] if i<12 else [40+i%8,1+i%3,10+i] for i in range(24)],dtype=float)
    with threadpool_limits(limits=1):synthetic,fit=fit_pipeline(train,'log_standard')
    synthetic.update(schema='frozen-rfm-v1',development_medians=np.median(train,axis=0).tolist(),provenance='Authored synthetic 24-row development set; seed13,n_init10,k2')
    write_json(demo/'model.json',synthetic)
    def csvfile(name,ids,values):
        with (demo/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['CustomerID','R','F','M']);w.writerows([[key,*x] for key,x in zip(ids,values)])
    csvfile('development.csv',[f'D{i:02}' for i in range(24)],train)
    csvfile('left.csv',[f'S{i:02}' for i in range(16)],train[:16])
    changed=train[4:20].copy();changed[0]=train[18];changed[1]=train[19]
    csvfile('right.csv',[f'S{i:02}' for i in range(4,20)],changed)
    dr,da=compare(demo/'left.csv',demo/'right.csv',demo/'model.json');export(dr,da,demo/'expected')
    write_json(public/'results.json',{'status':'completed','date':'2026-11-18','provider_report':report,
        'synthetic_report':dr,'synthetic_fit':fit,'cleaning':cleaning,'independent_scalar_rfm_max_abs_error':checks,
        'reference_prediction_rows_matched':len(prior),'api_cli_export_bytes_equal':True,
        'independent_recovered_pipeline_and_rule_checks':True,
        'source_model_sha256':digest(old_models.read_bytes()),'source_development_sha256':digest(development.read_bytes()),
        'provider_sha256':INPUT_SHA,'environment':{'python':platform.python_version(),'platform':platform.platform(),
          'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','openpyxl','matplotlib','threadpoolctl']}},
        'limits':'Known historical overlapping development/windows; same cutoff, not temporal migration or independent validation. Positive purchases exclude returns; no accuracy or causal claim.'})
    print(json.dumps({'status':'completed','accounting':report['accounting'],'methods':report['methods'],'synthetic':dr['accounting']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['provider','old-models','development','reference-predictions','audit','public']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();main(a.provider,a.old_models,a.development,a.reference_predictions,a.audit,a.public)
