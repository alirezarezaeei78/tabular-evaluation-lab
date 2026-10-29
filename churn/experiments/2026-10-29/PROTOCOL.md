# Feature-group refitting ablation — frozen before execution

Question: which declared feature groups contribute predictive value to the fixed
histogram-boosting model, conditional on the other supplied predictors?
Hypothesis: removing some groups may worsen AP; redundancy and refitting may also
produce unchanged or improved results. No winner or feature ranking is assumed.

Use the exact UCI Iranian Churn CSV hash and original eleven-predictor groups from
21 October. Retain the same grouped fit/calibration/evaluation partitions at seeds
13, 29 and 47; seed 13 is primary. These evaluation rows were already examined on
21 and 25 October: all new comparisons are exploratory, not independent validation.
The calibration partition is retained but unused. No temporal validation claim.
Status, Customer Value, Churn and identifiers remain excluded from inputs.

Freeze four disjoint groups, covering all eleven original features:

- Quality: Call Failure, Complains.
- Usage: Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers.
- Subscription: Subscription Length, Charge Amount, Tariff Plan.
- Age: Age Group, Age.

Fit the full eleven-feature model and four separate leave-one-group-out models
from scratch, each using identical original training rows and labels. Keep all
evaluation records and labels unchanged. Reuse the prior fixed HGB parameters:
learning_rate=0.06, max_iter=150, max_leaf_nodes=15, l2_regularization=1,
early_stopping=False, random_state=split seed; other scikit-learn 1.4.2 defaults.
HGB learns bin thresholds from fitting rows only; no external scaling or sampling.

References: training-label prevalence, plus a one-feature HGB with the same budget.
Select its one feature using maximum absolute Pearson correlation with the binary
target on fitting rows only, calculated by centered sums; constants score zero,
original feature order breaks ties. This intentionally simple training-derived
reference is not a tuned optimum. No calibration, tuning, early stopping or
selection from validation/evaluation outcomes. Fit all six models before scoring
the evaluation labels for a seed. All declared variants/seeds remain in the report.

Primary metric: average precision (AP). Secondary: Brier loss, and TP/FP/FN at an
exact ceil(0.10*N) review capacity with original row-ID tie breaking. Deltas compare
the same seed's matched records with its full-feature reference. Seeds overlap
in records and are sensitivity checks, not independent replications or confidence
intervals. Do not choose a new deployed feature set from this exposed analysis.

Group exact duplicates on the original eleven features before any ablation,
without changing the historical partitions. Removing columns may create new
feature-equivalent train/evaluation records; record those counts for every variant.
They are a limitation of the frozen matched-row comparison, not a clean reduced-
feature generalization test. Record feature counts, not unmeasured collection costs.

Check original split hashes against saved 21 October evidence, partition/group
separation, excluded columns, full-model numerical reproduction, finite scores,
independent tie-aware AP/Brier/capacity arithmetic, saved model replay and CSV/JSON
export parity. Save actual predictions, model files, versions, fit times/warnings,
source/split/code/protocol hashes. Use existing CPU, one thread, 20-minute total
budget; preserve any failure and never omit an unfavorable variant. Times are
operational records, not comparative runtime claims.

Data: Iranian Churn (2020), UCI, DOI 10.24432/C5JW3Z, CC BY 4.0. Public release:
authored code, protocol, aggregates, hashes and original measured PNG/SVG figure.
Raw provider records, fitted models and row-level evidence remain local. This is
a predictive model refit experiment, not causal importance, SHAP attribution,
evidence of retention benefit, or proof that a feature is operationally available.
