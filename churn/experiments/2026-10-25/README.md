# Class weighting versus normalized random oversampling

Executed local retrospective comparison, not a new algorithm or prospective test.
The21October grouped evaluation is deliberately reused and is exploratory. New
training interventions are applied to the same original partitions. See the
predeclared PROTOCOL.md and every seed in results.json; negative results remain.

From the repository root, using the existing churn requirements:

```sh
python -m pip install -r churn/requirements.txt
python -m unittest discover -s churn -p "test_*.py" -v
python churn/imbalance_experiment.py
python churn/render_imbalance.py
```

The provider CSV is fetched and hash-checked by the existing downloader. Raw data,
sampling indices and row predictions stay in churn/.local by default. An explicit
--audit-dir can point at another private local folder. No account/API key, paid
compute or hosted service is required. Public assets contain aggregate results.

Seed13: same631 records/101 labels; AP unweighted0.6831, weighted0.6741,
oversampled0.6788. Brier0.0834,0.1162,0.1135. At an exact64-record budget, true
positives are45,44,46. A0.5 cutoff instead finds37,86,86 with11,94,90 false
positives. Do not mistake a threshold-recall change for improved ranking or
better probability estimates. Both balancing variants had lower AP and higher
Brier in all three declared splits; seed variation is not a confidence interval.

Random oversampling adds only minority training rows with replacement. Constant
sample weights keep total weight equal to original training N. Standardization
uses the original training rows for all three strategies; C=1 and solver settings
are fixed. No resampling of evaluation, recalibration, threshold tuning or search.

Sources checked25October2026: [UCI Iranian Churn](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset),
DOI10.24432/C5JW3Z, donated2020, CC BY4.0;
[official scikit-learn1.4 LogisticRegression](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html),
class_weight/sample_weight and regularization conventions. Executed1.4.2; current
stable documentation is a different version. All original code/figures MIT; no
third-party code/artwork copied. Original provider attribution applies to data.
The features omit Status/Customer Value, IDs and target. Exact duplicate groups
never cross partitions. No causal retention, deployment or savings claim.
