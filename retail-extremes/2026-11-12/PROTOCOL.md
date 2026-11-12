# Frozen protocol: retail preprocessing and extreme-spend sensitivity

Declare before execution on12November2026. Research/analysis; no app or new
algorithm claim. The5November observation-window experiment already exposed this
benchmark. No independent holdout, causal customer-value or customer-type claim.

## Inputs and fixed cohort

UCI Online Retail, Chen2015, DOI10.24432/C5BW33, CC BY4.0. Excel SHA256
43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d.
Reuse the declared cleaning: finite positive integral customer IDs, valid naive
timestamps and invoices, non-C-prefixed invoices, finite quantity>0 and price>0.
Sequential exclusion counts must balance. Keep exact duplicate provider lines;
frequency counts distinct invoices. Monetary is gross positive-purchase pounds,
not net revenue/profit/lifetime value. Excluded returns are not subtracted.

RFM: recency in fractional days, distinct invoice frequency, monetary sum.
Development [2010-12-01,2011-07-01),2960customers. Comparison cutoff2011-12-01;
use the180day snapshot restricted to the same1664customers active in all30/90/180
windows from5November. Verify exact cohort hash and aggregation independently.
591matched customers were not in development. Windows overlap development;
this is historical conditional sensitivity, not a new validation split.

## Four fixed preprocessing pipelines and matched budgets

Compare raw RFM/StandardScaler, log1p(RFM)/StandardScaler (reference), raw
RFM/RobustScaler and log1p(RFM)/RobustScaler. Scalers fit development customers
only. RobustScaler centers on feature median and scales by25th–75th percentile
range, unit_variance=False. StandardScaler uses population variance. No clipping,
winsorization or outlier removal. Robust scaling does not make squared-distance
k-means an outlier-robust objective.

Hold k=2 from the known5November development-only choice; do not retune k for
these alternatives. Every KMeans uses random_state13,n_init10,max_iter300,tol1e-4,
algorithmlloyd,float64,CPU,thread limit1. Four clean fits. Keep those fitted
preprocessors and centers frozen for inference perturbations.

## Prespecified perturbations (not observed business events)

Frozen inference: select ceil(1%*1664)=17 comparison customers by original
monetary descending, tie by customer ID ascending. Multiply only their M by10;
R/F and every other row remain unchanged. On each fixed pipeline compare clean
versus stressed predictions: ARI, exact fixed-label change count, supports,
contingency and raw RFM profiles (medians only for groups of at least10). Check
all unmodified customers retain labels. These simulated values are not real spend.

Training influence: independently select ceil(1%*2960)=30 development customers
by the same original-M/tie rule and multiply only their M by10. Refit each entire
pipeline, including its scaler, with identical k/seed/budget. Predict unchanged
comparison rows; compare to its clean fit with permutation-invariant ARI and
optimal label-aligned agreement, supports and profiles. Raw label IDs from refits
have no shared semantic meaning. Four additional fits.

Sampling: choose80% of development customers without replacement, sample seeds
13,29,47; use the identical sampled indices across all four methods. Refit each
entire clean-data pipeline/scaler on that sample with KMeans seed13 fixed.
Predict the unchanged full matched comparison cohort. Save all three ARIs versus
the corresponding full-development fit, sample hashes and supports. Twelve fits;
no best-seed selection or confidence interval. Total20fixed-budget fits.

Also report all clean-method ARIs versus log+StandardScaler, and cluster sizes;
a stable giant cluster/singleton is not a useful-segmentation certificate.
No cross-geometry inertia/silhouette ranking or accuracy winner is inferred.

## Checks, records and limits

Independent scalar RFM sets/latest date/math.fsum; exact-rational contingency
ARI; brute-force label-permutation agreement for k2; direct median/IQR/mean/scale
arithmetic; direct squared-distance predictions/inertia; JSON model reload and
CSV full-precision round trips. Check identical IDs/order, train-only scaling,
selected-row counts, mutation locality, unchanged source arrays, zero/finiteness
and synthetic edge/tie cases. Keep all failed fits/warnings/negative outcomes.

Save provider/input/protocol/cohort/code hashes, configs,seeds, hardware,
dependency versions, actual stdout and all predictions/models privately. Public
release contains original code, aggregate metrics/supports/suppressed profiles,
primary-source/licence ledger and original measured PNG/SVG. Do not upload
transactions, IDs, per-customer predictions or fitted parameters. No recent-paper
claim: use current verified provider and pinned1.4.2 official method docs.

Existing hardware CPU;20minute model budget after input decode. Stop and preserve
incomplete work if exceeded. Observed elapsed time is audit info, not a speed claim.
Do not publish or queue LinkedIn during this preparation run. Native duplicate,
owner, public audience, image/alt and immutable public GitHub verification remain
publication gates. Static images only for this release; no video fallback.
