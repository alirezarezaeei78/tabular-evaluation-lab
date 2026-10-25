# Training imbalance strategies, declared before execution

Question: how do unweighted logistic regression, inverse-frequency class weighting
and training-only random oversampling change ranking, probability error and a
fixed review queue on the same UCI Iranian Churn rows?

Hypothesis: balancing may alter threshold errors without improving AP or Brier.
No winner is assumed. Primary metric: AP. Secondary: Brier, 0.5-cutoff errors and
precision/recall at exactly ceil(0.10*N) review capacity, using deterministic row-ID
ties. Dummy training prevalence is a reference. No causal retention inference.

Use the exact previous grouped splits at seeds13,29,47 (13 primary). Reused known
21October evaluation is exploratory, not fresh independent validation. The existing
11 predictors exclude Status, Customer Value, target and identifiers. Group exact
predictor duplicates; preserve all original development/evaluation records and
class supports. No temporal generalization claim; usable timestamps are absent.

Fit StandardScaler on original training rows only and reuse it across strategies.
All logistic fits use C=1, L2, lbfgs, tol=1e-4, max_iter=2000. Balanced weights use
N/(2*N_class). Oversampling keeps every training record and draws minority records
with replacement until class counts match, seed fixed per split. A constant
sample weight N_original/N_resampled keeps total weight equal to original N,
avoiding a regularization-strength change due solely to dataset expansion.
Calibration/development partition is preserved but not used for fitting or tuning.
No threshold tuning, recalibration, hyperparameter search or test-based selection.

Check group separation and exact hashes, train-only means, sampling indices and
weight mass, unchanged evaluation labels/supports, finite probabilities, independent
AP/Brier/count oracles and solver convergence. Save every seed, predictions,
sampling index, environment, stdout and negative outcome. Fits are CPU, one thread,
within20minutes. No runtime comparison. Public aggregates/plots only; row audits
and raw data stay local. Data: UCI Iranian Churn(2020), DOI10.24432/C5JW3Z, CC BY4.0.

New contribution relative to21October: explicit training-loss reweighting versus
normalized random duplication, rather than post-fit sigmoid calibration. Established
methods in a local experiment; no novelty, deployment or business-savings claim.
