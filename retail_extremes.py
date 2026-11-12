"""Declared preprocessing/perturbation sensitivity on locally fetched Online Retail."""
import argparse
from collections import Counter
from itertools import permutations
import hashlib
import importlib.metadata
import json
import math
import os
# Declared one-CPU budget; avoids optional Windows physical-core probing.
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from threadpoolctl import threadpool_limits
from retail_rfm import clean_transactions, rfm, scalar_rfm_check, exact_ari, direct_predict, profiles

METHODS = ('raw_standard', 'log_standard', 'raw_robust', 'log_robust')
INPUT_SHA = '43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d'
COHORT_SHA = '1ef8797f079395b7f5de5820f979b05eee14684f71554cf8291040676bf59449'
FIT_CONFIG = dict(n_clusters=2, random_state=13, n_init=10, max_iter=300, tol=1e-4, algorithm='lloyd')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Require finite nonnegative RFM rows, shape(n>=2,3)')
    return x


def prepare(raw, method):
    if method not in METHODS:
        raise ValueError('Unknown preprocessing method')
    x = validate(raw)
    return np.log1p(x) if method.startswith('log_') else x.copy()


def quantile_scalar(values, q):
    ordered = sorted(map(float, values))
    pos = (len(ordered)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def fit_pipeline(raw, method):
    x = prepare(raw, method)
    scaler = RobustScaler(quantile_range=(25,75), unit_variance=False) if method.endswith('_robust') else StandardScaler()
    z = scaler.fit_transform(x)
    if method.endswith('_robust'):
        center = np.array([quantile_scalar(x[:,j],.5) for j in range(3)])
        scale = np.array([quantile_scalar(x[:,j],.75)-quantile_scalar(x[:,j],.25) for j in range(3)])
    else:
        center = np.array([math.fsum(x[:,j])/len(x) for j in range(3)])
        scale = np.array([math.sqrt(math.fsum((float(v)-center[j])**2 for v in x[:,j])/len(x)) for j in range(3)])
    scale = np.where(scale < 10*np.finfo(float).eps, 1., scale)
    np.testing.assert_allclose(scaler.scale_, scale, rtol=1e-12, atol=1e-12)
    actual_center = scaler.center_ if method.endswith('_robust') else scaler.mean_
    np.testing.assert_allclose(actual_center, center, rtol=1e-12, atol=1e-12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = KMeans(**FIT_CONFIG).fit(z)
    direct = direct_predict(z, model.cluster_centers_)
    np.testing.assert_array_equal(direct, model.predict(z))
    inertia = math.fsum(float(np.dot(row-model.cluster_centers_[lab],row-model.cluster_centers_[lab])) for row,lab in zip(z,model.labels_))
    assert math.isclose(model.inertia_, inertia, rel_tol=1e-11, abs_tol=1e-8)
    bundle = {'method':method,'center':actual_center.tolist(),'scale':scaler.scale_.tolist(),'centers':model.cluster_centers_.tolist()}
    # Recovered parameters predict exactly as the fitted pipeline on its own rows.
    np.testing.assert_array_equal(predict(bundle,raw),model.predict(z))
    record = {'iterations':int(model.n_iter_),'inertia_within_own_geometry_only':float(model.inertia_),
              'development_supports':np.bincount(model.labels_,minlength=2).tolist(),
              'converged_before_max_iter':bool(model.n_iter_<FIT_CONFIG['max_iter']),
              'valid_two_groups':bool(len(set(model.labels_))==2),'warnings':[str(w.message) for w in caught]}
    return bundle,record


def predict(bundle, raw):
    x = prepare(raw,bundle['method'])
    z = (x-np.array(bundle['center']))/np.array(bundle['scale'])
    return direct_predict(z,bundle['centers'])


def stress_spend(frame, fraction=.01, multiplier=10.):
    if not 0<=fraction<=1 or not math.isfinite(multiplier) or multiplier<=0:
        raise ValueError('Invalid stress parameters')
    if not frame.index.is_unique:
        raise ValueError('Customer IDs must be unique')
    x = validate(frame.to_numpy())
    count = math.ceil(len(frame)*fraction)
    # Source index is sorted string IDs; explicit ID order breaks spend ties.
    positions = sorted(range(len(frame)),key=lambda i:(-float(x[i,2]),str(frame.index[i])))[:count]
    out = frame.copy(deep=True)
    out.iloc[positions,2] *= multiplier
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Stressed spend overflow')
    return out,np.array(positions,dtype=int)


def partition_comparison(a,b):
    a,b = np.asarray(a,dtype=int),np.asarray(b,dtype=int)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2:
        raise ValueError('Require equal nonempty partitions')
    value=float(adjusted_rand_score(a,b));assert math.isclose(value,exact_ari(a.tolist(),b.tolist()),abs_tol=1e-12)
    labels=sorted(set(a)|set(b))
    if len(labels)>6:
        raise ValueError('Brute-force alignment capped at6 labels')
    matched=max(sum(int(x==mapping[y]) for x,y in zip(a,b)) for perm in permutations(labels) for mapping in [dict(zip(labels,perm))])
    pair=Counter(zip(a.tolist(),b.tolist()))
    return {'ari':value,'optimal_label_aligned_agreement':matched/len(a),
            'contingency':[{'left':int(x),'right':int(y),'count':n} for (x,y),n in sorted(pair.items())]}


def main(data,audit,output,protocol):
    protocol=Path(protocol); frozen=json.loads(protocol.with_name('protocol-frozen.json').read_text(encoding='utf-8'))
    assert sha(protocol.read_bytes())==frozen['sha256']
    data=Path(data);assert sha(data.read_bytes())==INPUT_SHA
    frame=pd.read_excel(data,engine='openpyxl');start=time.perf_counter()
    clean,cleaning=clean_transactions(frame)
    dev,dev_rows=rfm(clean,'2010-12-01','2011-07-01')
    errors={'development':scalar_rfm_check(dev,dev_rows,'2011-07-01')}
    windows={}
    for days in [30,90,180]:
        f,rows=rfm(clean,pd.Timestamp('2011-12-01')-pd.Timedelta(days=days),'2011-12-01')
        windows[days]=f
        errors[str(days)]=scalar_rfm_check(f,rows,'2011-12-01')
    ids=sorted(set.intersection(*(set(f.index) for f in windows.values())))
    assert sha('\n'.join(ids).encode())==COHORT_SHA and len(ids)==1664 and len(dev)==2960
    comparison=windows[180].loc[ids].copy();dev_original=dev.copy(deep=True);comparison_original=comparison.copy(deep=True)
    stressed_dev,dev_positions=stress_spend(dev);stressed_comparison,eval_positions=stress_spend(comparison)
    assert len(dev_positions)==30 and len(eval_positions)==17
    audit=Path(audit);audit.mkdir(parents=True,exist_ok=True)
    dev.to_csv(audit/'development-rfm.csv',float_format='%.17g');comparison.to_csv(audit/'comparison-rfm.csv',float_format='%.17g')
    private=pd.DataFrame({'CustomerID':ids});models={};result={'date':'2026-11-12','status':'running','protocol_sha256':frozen['sha256'],
        'protocol_frozen_at':frozen['frozen_at'],'input_sha256':INPUT_SHA,'matched_customer_ids_sha256':COHORT_SHA,
        'data_attribution':'Chen,D.(2015). Online Retail. UCI. DOI10.24432/C5BW33. CC BY4.0.',
        'cleaning':cleaning,'development_customers':len(dev),'matched_customers':len(ids),
        'matched_new_to_development':len(set(ids)-set(dev.index)),'k':2,'fit_config':FIT_CONFIG,
        'preprocessing_methods':list(METHODS),'stress':{'multiplier':10,'fraction':.01,'development_affected':30,'comparison_affected':17,'definition':'Simulated monetary multiplication, R/F unchanged; not an observed business event'},
        'scalar_rfm_max_abs_error':errors,'methods':{},'fits':[],'sampling':[],
        'exposure':'Same known5November development/comparison customers. Overlapping historical windows; exploratory sensitivity, no independent validation or real customer-type labels.',
        'monetary_definition':'Gross positive-purchase pounds, excludes returns rather than netting; exact duplicate source lines retained.',
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'threads':1,
                       'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','openpyxl','threadpoolctl','matplotlib']}},
        'runtime_claim':False,'confidence_interval':None}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def fit(raw,method,kind):
        if time.perf_counter()-start>1200:raise TimeoutError('Declared model budget exceeded')
        bundle,record=fit_pipeline(raw,method);models[method+'_'+kind]=bundle
        result['fits'].append({'method':method,'condition':kind,**record});print(json.dumps({'fit':len(result['fits']),'method':method,'condition':kind,'supports':record['development_supports'],'warnings':record['warnings']}),flush=True)
        return bundle
    with threadpool_limits(limits=1):
        for method in METHODS:
            bundle=fit(dev,method,'clean');a=predict(bundle,comparison);b=predict(bundle,stressed_comparison)
            unaffected=np.ones(len(ids),dtype=bool);unaffected[eval_positions]=False;assert np.array_equal(a[unaffected],b[unaffected])
            reloaded=json.loads(json.dumps(bundle));assert np.array_equal(a,predict(reloaded,comparison))
            influenced=fit(stressed_dev,method,'training_stress');c=predict(influenced,comparison)
            record={'clean_supports':np.bincount(a,minlength=2).tolist(),
                    'clean_profiles':profiles(comparison,a),
                    'frozen_input_stress':{**partition_comparison(a,b),'changed_fixed_labels':int(np.sum(a!=b)),
                        'supports':np.bincount(b,minlength=2).tolist(),'profiles':profiles(stressed_comparison,b)},
                    'training_stress_refit':{**partition_comparison(a,c),'supports':np.bincount(c,minlength=2).tolist(),'profiles_on_clean_comparison':profiles(comparison,c)},
                    'sampling_aris':[]}
            result['methods'][method]=record;private[method+'_clean']=a;private[method+'_inference_stress']=b;private[method+'_training_stress']=c;save()
        for seed in [13,29,47]:
            indices=np.sort(np.random.default_rng(seed).choice(len(dev),math.floor(.8*len(dev)),replace=False));subset=dev.iloc[indices]
            sample_hash=sha('\n'.join(subset.index).encode())
            for method in METHODS:
                bundle=fit(subset,method,'sample_'+str(seed));a=private[method+'_clean'].to_numpy();b=predict(bundle,comparison)
                comp=partition_comparison(a,b);result['methods'][method]['sampling_aris'].append(comp['ari'])
                result['sampling'].append({'sample_seed':seed,'kmeans_seed':13,'method':method,'customers':len(subset),'sample_ids_sha256':sample_hash,**comp,'comparison_supports':np.bincount(b,minlength=2).tolist()})
                private[method+'_sample_'+str(seed)]=b;save()
    reference=private['log_standard_clean'].to_numpy()
    for method in METHODS:result['methods'][method]['clean_vs_log_standard']=partition_comparison(reference,private[method+'_clean'].to_numpy())
    pd.testing.assert_frame_equal(dev,dev_original);pd.testing.assert_frame_equal(comparison,comparison_original)
    assert len(result['fits'])==20
    private.to_csv(audit/'predictions.csv',index=False,float_format='%.17g');roundtrip=pd.read_csv(audit/'predictions.csv',dtype={'CustomerID':str},float_precision='round_trip');pd.testing.assert_frame_equal(private,roundtrip)
    (audit/'models.json').write_text(json.dumps(models,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    result.update(status='completed',completed_at=pd.Timestamp.now(tz='Asia/Tehran').isoformat(),wall_seconds_audit_only=time.perf_counter()-start,
                  model_fits_completed=20,private_prediction_rows=len(private),private_prediction_columns=len(private.columns)-1,
                  independent_metrics_scaler_rfm_and_distance_checks_passed=True,private_csv_roundtrip_exact=True,source_arrays_unchanged=True,
                  private_artifact_hashes={p.name:sha(p.read_bytes()) for p in audit.iterdir() if p.is_file()})
    save();print(json.dumps({'status':result['status'],'fits':20,'matched_customers':len(ids),'methods':result['methods']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--protocol',required=True,type=Path);a=p.parse_args();main(a.data,a.audit,a.output,a.protocol)
