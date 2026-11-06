"""Local model-backed input stress reports. Trusted local model bundles only."""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits

from churn_lab import FEATURES, exact_capacity, load_data, oracle_ap, split_indices

USAGE = ['Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers']
USAGE_INDEX = [FEATURES.index(f) for f in USAGE]
POLICIES = ('native', 'median', 'guarded')
PARAMS = dict(learning_rate=.06, max_iter=150, max_leaf_nodes=15,
              l2_regularization=1., early_stopping=False)
DATA_SHA = '90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False)+'\n'


def read_input(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > 1_000_000:
        raise ValueError('Input CSV must be at most1MB')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    names = reader.fieldnames
    if not names or len(names) != len(set(names)):
        raise ValueError('Unique nonempty CSV headers required')
    if not set(FEATURES) <= set(names) or set(names)-set(FEATURES+['record_id','Churn']):
        raise ValueError('Require eleven exact feature columns; only record_id/Churn are optional')
    values, keys, labels = [], [], []
    for index, row in enumerate(reader, 1):
        if index > 5000 or None in row or any(v is None for v in row.values()):
            raise ValueError('Require aligned rows, maximum5000')
        key = row.get('record_id', f'row-{index:06d}')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',key) or key in keys:
            raise ValueError('record_id must be unique, 1..64letters/digits/underscore/hyphen')
        current = []
        for feature in FEATURES:
            cell = row[feature].strip()
            if not cell:
                current.append(float('nan')); continue
            try:
                number = float(cell)
            except ValueError:
                raise ValueError(f'Row{index}: nonnumeric {feature}') from None
            if not math.isfinite(number):
                raise ValueError(f'Row{index}: nonfinite {feature}; blank means missing')
            current.append(number)
        if 'Churn' in names:
            if row['Churn'] not in ('0','1'):
                raise ValueError('Optional labels must be0or1')
            labels.append(int(row['Churn']))
        values.append(current); keys.append(key)
    if not values:
        raise ValueError('Input has no records')
    return np.asarray(values), keys, np.asarray(labels) if labels else None


def validate_bundle(bundle):
    if bundle.get('features') != FEATURES or bundle.get('schema') != 'churn-stress-v1':
        raise ValueError('Incompatible trusted model bundle')
    for name in ('minimum','maximum','median'):
        value = np.asarray(bundle[name])
        if value.shape != (len(FEATURES),) or not np.isfinite(value).all():
            raise ValueError('Invalid fitted feature statistics')
    if not np.all(np.asarray(bundle['minimum']) <= bundle['maximum']):
        raise ValueError('Reversed fitted ranges')
    if not 0 <= bundle['prior'] <= 1 or not np.isfinite(bundle['prior']):
        raise ValueError('Invalid fitting prevalence')


def score_array(values, ids, bundle, policy):
    validate_bundle(bundle)
    values = np.asarray(values, dtype=float)
    if policy not in POLICIES:
        raise ValueError('Unknown policy')
    if values.shape != (len(ids), len(FEATURES)) or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Aligned nonempty unique IDs required')
    if np.isinf(values).any():
        raise ValueError('Infinite input not accepted')
    missing = np.isnan(values)
    outside = (~missing) & ((values < bundle['minimum']) | (values > bundle['maximum']))
    accepted = ~(missing | outside).any(axis=1) if policy == 'guarded' else np.ones(len(ids),dtype=bool)
    x = np.where(missing,bundle['median'],values) if policy == 'median' else values.copy()
    model_scores = np.full(len(ids),np.nan)
    if accepted.any():
        with threadpool_limits(limits=1):
            model_scores[accepted] = bundle['model'].predict_proba(x[accepted])[:,1]
    effective = np.where(accepted,model_scores,bundle['prior'])
    if not np.isfinite(effective).all() or ((effective < 0)|(effective > 1)).any():
        raise ValueError('Model emitted invalid probabilities')
    chosen = exact_capacity(effective,ids,.1)
    rows=[]
    for i,key in enumerate(ids):
        flags=[f'missing:{f}' for j,f in enumerate(FEATURES) if missing[i,j]]
        flags += [f'outside_fit_range:{f}' for j,f in enumerate(FEATURES) if outside[i,j]]
        rows.append({'record_id':key,'status':'model_scored' if accepted[i] else 'rejected_prior_fallback',
            'input_flags':json.dumps(flags,separators=(',',':')),
            'model_probability':float(model_scores[i]) if accepted[i] else None,
            'fallback_probability':None if accepted[i] else float(bundle['prior']),
            'effective_policy_score':float(effective[i]),'exact_10_percent_review':bool(chosen[i])})
    return {'schema':'churn-stress-report-v1','policy':policy,'model_seed':int(bundle['seed']),
        'n':len(ids),'accepted':int(accepted.sum()),'rejected':int((~accepted).sum()),
        'coverage':float(accepted.mean()),'exact_capacity_count':int(chosen.sum()),
        'interpretation':'Input range checks are not reliability certification. Prior fallback is not an individualized model estimate.','rows':rows}


def csv_text(report):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(report['rows'][0]),lineterminator='\n')
    writer.writeheader();writer.writerows(report['rows'])
    return stream.getvalue()


def write_outputs(report,json_path,csv_path,protected=()):
    paths=[Path(json_path),Path(csv_path)]
    resolved=[path.resolve() for path in paths]
    if len(set(resolved))!=2 or any(path in {Path(f).resolve() for f in protected} for path in resolved):
        raise ValueError('Outputs must be distinct and cannot replace input/model')
    if any(path.exists() for path in paths):
        raise FileExistsError('Preserve existing evidence; choose new output names')
    for path,content in zip(paths,[json_text(report),csv_text(report)]):
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(content)


def metrics(y,scores,ids):
    y=np.asarray(y); scores=np.asarray(scores)
    if not len(y):
        return {'n':0,'positives':0,'ap':None,'brier':None,'capacity':None}
    ap=float(average_precision_score(y,scores)) if len(set(y))==2 else None
    brier=float(brier_score_loss(y,scores))
    assert ap is None or math.isclose(ap,oracle_ap(y,scores),abs_tol=1e-12)
    assert math.isclose(brier,math.fsum((float(a)-float(b))**2 for a,b in zip(y,scores))/len(y),abs_tol=1e-12)
    flag=exact_capacity(scores,ids,.1); k=math.ceil(.1*len(y))
    expected=sorted(range(len(y)),key=lambda i:(-float(scores[i]),ids[i]))[:k]
    assert set(np.flatnonzero(flag))==set(expected)
    tp=int(y[flag].sum());fp=int(flag.sum())-tp;fn=int(y.sum())-tp
    return {'n':len(y),'positives':int(y.sum()),'ap':ap,'brier':brier,
        'capacity':{'selected':int(flag.sum()),'tp':tp,'fp':fp,'fn':fn,
                    'precision':tp/int(flag.sum()) if flag.any() else None,
                    'recall':tp/int(y.sum()) if y.sum() else None}}


def run(data,previous,protocol,output,audit):
    output,audit,protocol=Path(output),Path(audit),Path(protocol)
    if output.exists() or audit.exists():
        raise FileExistsError('Use new result/audit directories')
    frozen=json.loads(protocol.with_name('protocol-frozen.json').read_text())
    assert sha(protocol.read_bytes())==frozen['protocol_sha256']
    assert sha(Path(data).read_bytes())==DATA_SHA
    frame,x,y,groups=load_data(data)
    old=json.loads(Path(previous).read_text(encoding='utf-8-sig'))
    output.mkdir();audit.mkdir();started=time.monotonic()
    result={'status':'running','protocol_sha256':frozen['protocol_sha256'],
        'frozen_at':frozen['frozen_at'],'data_sha256':DATA_SHA,'rows':len(frame),
        'features':FEATURES,'excluded':['Status','Customer Value','Churn','record_id'],
        'config':PARAMS,'scope':'Simulated input perturbations on repeated exposed grouped evaluations; exploratory, not production drift frequency or causal retention benefit.',
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'cpu':platform.processor(),'threads':1,'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib','threadpoolctl','matplotlib']}},'runs':[]}
    with threadpool_limits(limits=1):
        for seed in (13,29,47):
            parts=split_indices(x,y,groups,seed); fit,test=parts['fit'],parts['evaluation']
            ref=next(r for r in old['runs'] if r['seed']==seed)
            partitions={k:{'rows':len(v),'positives':int(y[v].sum()),'sha256':sha(np.sort(v).astype('<i8').tobytes())} for k,v in parts.items()}
            assert all(v['sha256']==ref['partitions'][k]['index_sha256'] for k,v in partitions.items())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model=HistGradientBoostingClassifier(**PARAMS,random_state=seed).fit(x[fit],y[fit])
            assert model.n_iter_==150
            bundle={'schema':'churn-stress-v1','features':FEATURES,'model':model,'seed':seed,
                'minimum':x[fit].min(axis=0),'maximum':x[fit].max(axis=0),
                'median':np.median(x[fit],axis=0),'prior':float(y[fit].mean())}
            bundle_path=audit/f'model-{seed}.joblib';joblib.dump(bundle,bundle_path)
            loaded=joblib.load(bundle_path);np.testing.assert_array_equal(model.predict_proba(x[test]),loaded['model'].predict_proba(x[test]))
            ids=[f'source-{i:06d}' for i in test]; base=metrics(y[test],model.predict_proba(x[test])[:,1],ids)
            for key in ('ap','brier'):assert math.isclose(base[key],ref['models']['boosting'][key],abs_tol=1e-12)
            all_missing=x[test].copy();all_missing[:,USAGE_INDEX]=np.nan
            indices=np.sort(np.random.default_rng(seed+900).choice(len(test),math.floor(.25*len(test)),replace=False))
            some_missing=x[test].copy();some_missing[np.ix_(indices,USAGE_INDEX)]=np.nan
            shifted=x[test].copy();shifted[:,USAGE_INDEX]*=2
            conditions={'clean':x[test],'usage_missing_all':all_missing,'usage_missing_25pct':some_missing,'usage_times2':shifted}
            run_result={'seed':seed,'partitions':partitions,'fit_warnings':[str(w.message) for w in caught],
                'completed_iterations':int(model.n_iter_),'missing_subset_rows':len(indices),
                'missing_subset_source_ids_sha256':sha('\n'.join(ids[i] for i in indices).encode()),
                'training_prior':bundle['prior'],'clean_reference':base,
                'constant_prior_reference':metrics(y[test],np.full(len(test),bundle['prior']),ids),'conditions':[]}
            for condition,values in conditions.items():
                for policy in POLICIES:
                    report=score_array(values,ids,loaded,policy)
                    scores=np.array([r['effective_policy_score'] for r in report['rows']])
                    accepted=np.array([r['model_probability'] is not None for r in report['rows']])
                    record={'condition':condition,'policy':policy,'coverage':report['coverage'],
                        'accepted':report['accepted'],'rejected':report['rejected'],
                        'rejected_positives':int(y[test][~accepted].sum()),
                        'all_row_policy_metrics':metrics(y[test],scores,ids),
                        'accepted_only_model_metrics':metrics(y[test][accepted],scores[accepted],[key for key,use in zip(ids,accepted) if use])}
                    run_result['conditions'].append(record)
                    report['rows']=[{**row,'observed_churn':int(label)} for row,label in zip(report['rows'],y[test])]
                    write_outputs(report,audit/f'{seed}-{condition}-{policy}.json',audit/f'{seed}-{condition}-{policy}.csv')
                    print(json.dumps({'seed':seed,**record}),flush=True)
            result['runs'].append(run_result)
            if time.monotonic()-started>1200:raise TimeoutError('20minute compute budget exceeded')
    result.update(status='completed',new_fits=3,conditions_per_seed=12,
        all_original_partition_hashes_match=True,all_clean_booster_metrics_reproduced=True,
        model_reload_exact=True,independent_metrics_passed=True,
        wall_seconds_observed=time.monotonic()-started,
        private_audit_sha256={f.name:sha(f.read_bytes()) for f in sorted(audit.iterdir()) if f.is_file()})
    (output/'results.json').write_text(json_text(result),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    experiment=commands.add_parser('evaluate')
    for arg in ('data','previous','protocol','output','audit'):experiment.add_argument('--'+arg,required=True)
    score=commands.add_parser('score')
    for arg in ('input','bundle','json','csv'):score.add_argument('--'+arg,required=True)
    score.add_argument('--policy',choices=POLICIES,default='guarded')
    args=parser.parse_args()
    if args.command=='evaluate':
        run(args.data,args.previous,args.protocol,args.output,args.audit)
    else:
        # joblib unpickling executes code: only use your own locally reproduced bundle.
        values,ids,labels=read_input(Path(args.input).read_text(encoding='utf-8-sig'))
        report=score_array(values,ids,joblib.load(args.bundle),args.policy)
        if labels is not None:report['descriptive_errors']=metrics(labels,np.array([r['effective_policy_score'] for r in report['rows']]),ids)
        write_outputs(report,args.json,args.csv,protected=[args.input,args.bundle])
        print(json.dumps({key:report[key] for key in ('policy','n','accepted','rejected','coverage','exact_capacity_count')}))


if __name__=='__main__':main()
