"""Executed synthetic teaching example; no customer data or benchmark accuracy claim."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def inputs(points_per_arc=120, seed=13):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, points_per_arc)
    a = np.column_stack((np.cos(t),np.sin(t)))
    b = np.column_stack((1-np.cos(t),.5-np.sin(t)))
    arcs = np.vstack((a,b)) + rng.normal(0,.035,size=(2*points_per_arc,2))
    return np.vstack((arcs,[[-1.8,-1.4],[-1.8,1.8],[2.8,-1.4],[2.8,1.8]]))


def validate(x):
    x = np.asarray(x,dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Require at least two finite rows and exactly two features')
    return x


def fit(x):
    x = validate(x)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    km = KMeans(n_clusters=2,n_init=10,random_state=13,algorithm='lloyd',max_iter=300,tol=1e-4).fit(z)
    db = DBSCAN(eps=.22,min_samples=5,metric='euclidean',algorithm='brute').fit(z)
    return scaler,z,km,db


def density_oracle(z, labels, core_indices, eps=.22, min_samples=5):
    """Independent dense distances, BFS of core graph, and border/noise checks."""
    distances = np.sqrt(np.sum((z[:,None,:]-z[None,:,:])**2,axis=2))
    near = distances <= eps
    core = near.sum(axis=1) >= min_samples
    assert set(np.flatnonzero(core)) == set(core_indices)
    components, unseen = [], set(np.flatnonzero(core))
    while unseen:
        start = min(unseen);unseen.remove(start);group={start};todo=[start]
        while todo:
            i=todo.pop()
            connected=set(np.flatnonzero(near[i]&core)) & unseen
            unseen-=connected;group|=connected;todo.extend(sorted(connected))
        components.append(group)
    group_labels=[]
    for group in components:
        assigned={int(labels[i]) for i in group}
        assert len(assigned)==1 and -1 not in assigned
        group_labels.append(next(iter(assigned)))
    assert len(set(group_labels)) == len(group_labels)
    for i in np.flatnonzero(~core):
        reachable={int(labels[j]) for j in np.flatnonzero(near[i]&core)}
        assert (labels[i] in reachable) if reachable else labels[i]==-1
    return {'core_components':len(components),'core_rows':int(core.sum()),
        'border_rows':int(np.sum((~core)&(labels!=-1))), 'noise_rows':int(np.sum(labels==-1)),
        'eps_counts_include_self':bool(np.all(np.diag(near))), 'independent_graph_checks':True}


def run(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    x=inputs();probes=inputs(40,29)
    scaler,z,km,db=fit(x)
    dist2=np.sum((z[:,None,:]-km.cluster_centers_[None,:,:])**2,axis=2)
    np.testing.assert_array_equal(np.argmin(dist2,axis=1),km.labels_)
    inertia=math.fsum(float(dist2[i,km.labels_[i]]) for i in range(len(z)))
    assert math.isclose(inertia,km.inertia_,rel_tol=1e-12)
    density=density_oracle(z,db.labels_,db.core_sample_indices_)
    changed=probes*1e6+1e8
    before=(scaler.mean_.copy(),scaler.scale_.copy(),km.cluster_centers_.copy())
    scaler.transform(changed)
    for a,b in zip(before,(scaler.mean_,scaler.scale_,km.cluster_centers_)):np.testing.assert_array_equal(a,b)
    unit_x=x*np.array([100.,1.]);unit_scaler=StandardScaler().fit(unit_x)
    np.testing.assert_allclose(unit_scaler.transform(unit_x),z,rtol=1e-12,atol=1e-12)
    short=math.hypot(.1,.1);changed_distance=math.hypot(10.,.1)
    assert short < .22 < changed_distance
    np.testing.assert_allclose(scaler.mean_,np.mean(x,axis=0),atol=1e-14)
    config={'kmeans':km.get_params(),'dbscan':db.get_params(),'fit_rows':len(x),'probe_rows':len(probes),
        'generator':'Authored noisy semicircles + four isolated points; code in inputs()',
        'fit_generator_seed':13,'probe_generator_seed':29,'probe_points_per_arc':40,'noise_std':.035,
        'preprocessing':'StandardScaler fit only on the244 fitting rows; probes transformed only'}
    result={'scope':'Established teaching on authored synthetic inputs, not real customer segmentation or a comparative accuracy benchmark.',
        'config':config,'density':density,'kmeans':{'groups':int(len(set(km.labels_))),'noise_rows':0,
            'iterations':int(km.n_iter_),'inertia':float(km.inertia_),'independent_inertia':inertia,
            'centers_scaled':km.cluster_centers_.tolist(),'nearest_center_check':True},
        'scaler':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'samples_seen':int(scaler.n_samples_seen_)},
        'distance_example':{'original':short,'after_x_units_times_100':changed_distance,'eps':.22},
        'checks':{'held_out_transform_does_not_change_fit':True,'positive_unit_change_standardization_parity':True,
            'scaler_matches_independent_mean':True,'density_graph_oracle':True,'dbscan_has_builtin_predict':hasattr(db,'predict')},
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
            'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','matplotlib']}},
        'quality_metric_claimed':False,'novelty_claimed':False,'runtime_claimed':False,'hyperparameter_search':False}
    assert not result['checks']['dbscan_has_builtin_predict']
    for name,data,columns in [('fit-points.csv',np.column_stack((x,z,km.labels_,db.labels_)),['x1','x2','scaled_x1','scaled_x2','kmeans_label','dbscan_label']),
                              ('probes.csv',np.column_stack((probes,scaler.transform(probes),km.predict(scaler.transform(probes)))),['x1','x2','scaled_x1','scaled_x2','kmeans_prediction'])]:
        np.savetxt(out/name,data,delimiter=',',header=','.join(columns),comments='',fmt='%.17g')
        np.testing.assert_allclose(np.loadtxt(out/name,delimiter=',',skiprows=1),data,rtol=0,atol=0)
    result['input_output_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['fit-points.csv','probes.csv']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='outputs');args=parser.parse_args();run(args.output)
