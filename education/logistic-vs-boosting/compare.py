"""Executed synthetic illustration of binary scoring rules, with independent checks."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def independent_ap(y, p):
    y, p = np.asarray(y), np.asarray(p)
    positives = int(y.sum())
    if not positives:
        return 0.0
    previous, value = 0.0, 0.0
    for threshold in sorted(set(p.tolist()), reverse=True):
        chosen = p >= threshold
        tp = int(y[chosen].sum())
        recall = tp / positives
        value += (recall - previous) * tp / int(chosen.sum())
        previous = recall
    return value


def models():
    return (
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        GradientBoostingClassifier(loss='log_loss', n_estimators=30, learning_rate=.1,
                                   max_depth=2, random_state=13),
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    x, y = make_moons(n_samples=600, noise=.18, random_state=13)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=29)
    lr, gb = models()
    lr.fit(x[train], y[train])
    gb.fit(x[train], y[train])
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append({'name': name, 'passed': True})

    check('partition disjoint and complete', not set(train) & set(test) and set(train) | set(test) == set(range(600)))
    check('scaler training-only mean and count', np.allclose(lr[0].mean_, x[train].mean(0)) and lr[0].n_samples_seen_ == len(train))
    probabilities = {name: model.predict_proba(x[test]) for name, model in [('logistic', lr), ('boosting', gb)]}
    check('finite normalized probabilities', all(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1) for p in probabilities.values()))
    check('binary class mapping', all(model.classes_.tolist() == [0, 1] for model in [lr, gb]))
    check('sigmoid of linear score matches probability', np.allclose(expit(lr.decision_function(x[test])), probabilities['logistic'][:, 1], atol=1e-12))
    probes = np.array([[0., 0.], [.3, 0.], [0., .8], [.3, .8]])
    z = lr.decision_function(probes)
    check('linear logit has zero cross-interaction', abs(z[3] - z[2] - z[1] + z[0]) < 1e-12)
    prior = gb.init_.predict_proba(x[test])[:, 1]
    reconstructed = np.log(prior / (1 - prior))
    for tree in gb.estimators_[:, 0]:
        reconstructed += gb.learning_rate * tree.predict(x[test])
    check('additive tree leaf scores reconstruct final logits', np.allclose(reconstructed, gb.decision_function(x[test]), atol=1e-12))
    stages = list(gb.staged_predict_proba(x[test]))
    check('last staged probabilities match final predictions', len(stages) == 30 and np.allclose(stages[-1], probabilities['boosting'], atol=1e-12))
    check('independent AP including tied scores', all(abs(independent_ap(y[test], p[:, 1]) - average_precision_score(y[test], p[:, 1])) < 1e-12 for p in probabilities.values()) and abs(independent_ap([1,0,1,0],[.5,.5,.2,.1]) - (1/4 + 1/3)) < 1e-12)
    lr2, gb2 = models()
    lr2.fit(x[train], y[train]); gb2.fit(x[train], y[train])
    check('same-seed rerun probabilities match', np.array_equal(lr2.predict_proba(x[test]), probabilities['logistic']) and np.array_equal(gb2.predict_proba(x[test]), probabilities['boosting']))
    rejected = []
    for model in [lr, gb]:
        try:
            model.predict_proba(np.zeros((1, 3)))
        except ValueError:
            rejected.append(True)
    check('wrong feature count rejected by both models', len(rejected) == 2)
    gx = np.linspace(-1.6, 2.7, 96); gy = np.linspace(-1.2, 1.7, 76)
    xx, yy = np.meshgrid(gx, gy); grid = np.c_[xx.ravel(), yy.ravel()]
    steps = [1, 5, 10, 20, 30]
    all_grid = list(gb.staged_predict_proba(grid))
    gb_grid = np.array([all_grid[i-1][:, 1].reshape(xx.shape) for i in steps])
    lr_grid = lr.predict_proba(grid)[:, 1].reshape(xx.shape)
    check('rendered final grid matches actual model', np.array_equal(gb_grid[-1].ravel(), gb.predict_proba(grid)[:, 1]) and np.array_equal(lr_grid.ravel(), lr.predict_proba(grid)[:, 1]))
    with (output/'data.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['row','x1','x2','label','partition'])
        tr = set(train)
        w.writerows((i, float(v[0]), float(v[1]), int(y[i]), 'train' if i in tr else 'held_out') for i,v in enumerate(x))
    with (output/'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['row','label','logistic_probability','boosting_probability'])
        w.writerows((int(row), int(y[row]), float(probabilities['logistic'][i,1]), float(probabilities['boosting'][i,1])) for i,row in enumerate(test))
    np.savez_compressed(output/'render-data.npz', gx=gx, gy=gy, logistic=lr_grid, boosting=gb_grid, stages=np.array(steps), test_x=x[test], test_y=y[test])
    summary = {'purpose':'Educational synthetic illustration, not a churn/generalization benchmark', 'python':platform.python_version(),
               'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','matplotlib','Pillow','imageio-ffmpeg']},
               'data':{'generator':'sklearn.make_moons','samples':600,'noise':.18,'seed':13,'split_seed':29,'train':len(train),'held_out':len(test),'positive_held_out':int(y[test].sum()),'selection':'No tuning, one prespecified split'},
               'logistic':{'C':1.,'max_iter':1000,'coefficients_after_scaling':lr[1].coef_.tolist(),'intercept':lr[1].intercept_.tolist(),'iterations':lr[1].n_iter_.tolist()},
               'boosting':{'loss':'log_loss','trees':30,'max_depth':2,'learning_rate':.1,'seed':13,'rendered_stages':steps},
               'metrics':{name:{'average_precision':float(average_precision_score(y[test],a[:,1])),'brier':float(brier_score_loss(y[test],a[:,1])),'log_loss':float(log_loss(y[test],a))} for name,a in probabilities.items()},
               'checks':checks,'checks_passed':len(checks), 'data_sha256':hashlib.sha256((output/'data.csv').read_bytes()).hexdigest(),
               'command':'python compare.py --output outputs', 'limits':['Synthetic nonlinear task chosen for teaching; not representative customer data.','No hyperparameter search, statistical interval, speed or superiority claim.','Probability outputs are not certified calibrated confidence.']}
    (output/'results.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True); run(ap.parse_args().output)
