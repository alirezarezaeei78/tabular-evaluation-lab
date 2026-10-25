"""Matched original-row evaluation of three declared training strategies."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits
from churn_lab import load_data, split_indices, evaluate, sha256, write_json
from run_benchmark import fetch_data, DATA_SHA

HERE=Path(__file__).resolve().parent
OUT=HERE/'experiments/2026-10-25'


def oversample_indices(y, seed):
    y=np.asarray(y)
    if y.ndim!=1 or set(y)!={0,1}:raise ValueError('Both binary classes required')
    counts=np.bincount(y);minority=int(np.argmin(counts))
    rng=np.random.default_rng(seed)
    extra=rng.choice(np.flatnonzero(y==minority),size=int(counts.max()-counts.min()),replace=True)
    indices=np.concatenate([np.arange(len(y)),extra])
    return indices,np.full(len(indices),len(y)/len(indices))


def run(output=OUT,audit=HERE/'.local/imbalance-audit'):
    output.mkdir(parents=True,exist_ok=True);audit.mkdir(parents=True,exist_ok=True)
    source=fetch_data();_,x,y,groups=load_data(source);assert sha256(source.read_bytes())==DATA_SHA
    summary={'scope':'Exploratory reuse of21October grouped evaluation; no independent validation',
      'protocol_sha256':sha256((OUT/'PROTOCOL.md').read_bytes()),'data_sha256':DATA_SHA,
      'data_citation':'Iranian Churn(2020),UCI,DOI10.24432/C5JW3Z,CC BY4.0',
      'environment':{'python':platform.python_version(),'cpu':platform.processor(),'fit_threads':1,
      'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','matplotlib']}},'runs':[]}
    old=json.loads((HERE/'experiments/2026-10-21/results.json').read_text())
    with threadpool_limits(limits=1):
      for seed in [13,29,47]:
        parts=split_indices(x,y,groups,seed);fit=parts['fit'];test=parts['evaluation']
        scaler=StandardScaler().fit(x[fit]);assert scaler.n_samples_seen_==len(fit)
        np.testing.assert_allclose(scaler.mean_,x[fit].mean(0),atol=1e-12)
        xf=scaler.transform(x[fit]);xt=scaler.transform(x[test]);yf=y[fit]
        sample,weights=oversample_indices(yf,seed)
        assert np.all(sample<len(fit)) and set(sample)==set(range(len(fit)))
        assert np.bincount(yf[sample])[0]==np.bincount(yf[sample])[1]
        assert math.isclose(weights.sum(),len(fit),abs_tol=1e-9)
        ids=[f'source-{i:06d}' for i in test]
        run={'seed':seed,'partitions':{k:{'rows':len(v),'positives':int(y[v].sum()),'index_sha256':sha256(np.sort(v).astype('<i8').tobytes())} for k,v in parts.items()},'models':{},'fit_logs':{},'sampling':{'original_rows':len(fit),'resampled_rows':len(sample),'class_counts':np.bincount(yf[sample]).tolist(),'weight_sum':float(weights.sum())}}
        reference=next(v for v in old['runs'] if v['seed']==seed)
        for k in parts:assert run['partitions'][k]['index_sha256']==reference['partitions'][k]['index_sha256']
        predictions=pd.DataFrame({'source_row_zero_based':test,'y':y[test]})
        for name in ['prior','unweighted','balanced','oversampled']:
          if name=='prior':prob=np.full(len(test),yf.mean());iterations=0
          else:
            model=LogisticRegression(C=1,max_iter=2000,tol=1e-4,solver='lbfgs',class_weight='balanced' if name=='balanced' else None)
            with warnings.catch_warnings(record=True) as caught:
              warnings.simplefilter('always')
              model.fit(xf[sample] if name=='oversampled' else xf,yf[sample] if name=='oversampled' else yf,sample_weight=weights if name=='oversampled' else None)
            if caught:raise RuntimeError(str(caught[0].message))
            assert np.max(model.n_iter_)<2000;iterations=int(model.n_iter_[0]);prob=model.predict_proba(xt)[:,1]
          result=evaluate(y[test],prob,ids,.5)
          flag=prob>=.5;policy=result['policies']['cutoff_0.5'];assert policy['tp']==sum(int(a==1 and b) for a,b in zip(y[test],flag))
          selected=sorted(range(len(prob)),key=lambda i:(-float(prob[i]),ids[i]))[:math.ceil(.1*len(prob))]
          assert result['policies']['exact_10_percent']['tp']==sum(int(y[test][i]) for i in selected)
          pr,re,_=precision_recall_curve(y[test],prob);result['pr_curve']={'precision':pr.tolist(),'recall':re.tolist()}
          run['models'][name]=result;run['fit_logs'][name]={'converged':True,'iterations':iterations,'warnings':[]};predictions[name]=prob
        assert abs(run['models']['unweighted']['ap']-reference['models']['logistic']['ap'])<1e-10
        assert abs(run['models']['unweighted']['brier']-reference['models']['logistic']['brier'])<1e-10
        predictions.to_csv(audit/f'predictions-{seed}.csv',index=False,float_format='%.17g')
        np.savez(audit/f'sampling-{seed}.npz',source_training_rows=fit,sample_positions=sample,weights=weights)
        run['audit_hashes']={n:sha256((audit/n).read_bytes()) for n in [f'predictions-{seed}.csv',f'sampling-{seed}.npz']}
        summary['runs'].append(run);print(json.dumps({'seed':seed,'models':{n:{k:v[k] for k in ['ap','brier']} for n,v in run['models'].items()}}),flush=True)
    summary['checks']={'same_prior_exposed_evaluation_hashes':True,'prior_logistic_reference_reproduced':True,'train_only_scaling_sampling':True,'normalized_weight_mass':True,'independent_ap_brier_capacity_oracles':True,'all_three_seeds_retained':True}
    write_json(output/'results.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-dir',type=Path,default=HERE/'.local/imbalance-audit');args=parser.parse_args();run(audit=args.audit_dir)
