# Frozen missing-input stress report for Churn Decision Lab

Declared 6 November 2026 after schema/hash checks and before fitting or inspecting
new stress outcomes. New app/CLI feature, not a new algorithm or independent test.

Use the same public UCI Iranian Churn CSV SHA256
90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1,
3,150 records, CC BY 4.0, DOI 10.24432/C5JW3Z. Keep provider rows, row-level
predictions and fitted models local; code/hashes/aggregates only in public release.
The original eleven-predictor exact-duplicate groups and GroupShuffleSplit
partitions at seeds13,29,47 remain unchanged from21October. Status, Customer Value,
Churn and IDs are never model predictors. Check partition hashes and clean booster
AP/Brier against21October results. Exposed evaluation is exploratory. No timestamps
support prospective deployment evaluation; seeds overlap, not independent trials.

Fit one HistGradientBoostingClassifier per original fitting partition with the
original learning_rate=.06,max_iter=150,max_leaf_nodes=15,l2_regularization=1,
early_stopping=False,random_state=split seed and scikit-learn1.4.2 defaults.
No calibration, tuning, retraining on corruption or evaluation-based selection.
Compute per-feature fitting medians/minima/maxima and training-label prevalence
using fitting records only. The calibration partition stays separate and unused.

Preset four matched evaluation input conditions per seed:
1. Clean original predictors.
2. All four usage fields absent (Seconds of Use, Frequency of use, Frequency of
   SMS, Distinct Called Numbers) on every record, represented as NaN. Their absence
   is simulated and is not an observed production outage.
3. Those four usage fields absent on an independently drawn25percent of records,
   chosen once by numpy.default_rng(seed+900).choice without replacement; use
   floor(.25*N). Record selection hashes. Selection never uses labels.
4. Multiply all four usage fields by2 on all records; no clipping or rounding.
   This is a simulated unit/scale mismatch, not a claim about actual drift rates.

For each condition compare three frozen policies on exactly the same records:
- native: original clean-fit HGB with its actual built-in NaN routing;
- median: same model after missing cells alone are filled by fitting medians;
- guarded: reject missing, nonfinite or values outside fitting min/max, emitting
  explicit reason codes. Otherwise use original model. Rejected records have
  probability null. Fallback probability is training prevalence, separately
  labelled; it is not an individualized model probability.

For guarded evaluation report both accepted-only model AP/Brier and coverage,
supports/positive counts, and all-row prior-fallback AP/Brier. Never compare
accepted-only metrics as though evaluated on the full matched cohort. Baseline is
clean native policy and constant fitting-prevalence predictor on all rows. A value
inside training extrema is only a range check, not a certificate of reliability.
Missingness and out-of-range flags need not detect all scale mismatches.

Primary metrics AP and Brier on matched all-row policy outputs; secondary exact
ceil(.10*N) review TP/FP/FN/precision/recall, stable original-row-ID tie breaking,
coverage/abstention and accepted/rejected class support. All outcomes/conditions
retained. No positive result assumed, no causal retention gain or drift-frequency
claim. AP and Brier use independent scalar oracles; audit selected counts and
accepted-only denominators independently. Empty/one-class subsets report null AP
where undefined rather than inventing accuracy. Confidence intervals not computed.

Feature: documented local CLI/API takes a bounded CSV with exact eleven columns,
optional unique record_id and optional labels; empty cells mean missing. Unknown,
duplicate/ragged headers, empty input, nonnumeric text, nonfinite values and missing
columns have explicit errors. Default authored synthetic demo is separate from
provider data. A user loads a locally reproduced trusted model bundle, chooses
native/median/guarded policy, sees per-record input flags, model probability and
separately labelled fallback, then exports matching JSON/CSV. No paid key or hosted
demo is required. Joblib is trusted-local-only; never load arbitrary downloaded
model files. CLI must protect existing outputs and never overwrite input/model.

Before release execute safe demo through actual CLI, schema/empty/error paths,
guarded missing/out-of-range behavior, baseline comparisons, label exclusion,
partition separation, model reload, exact JSON/CSV parity including null scores
and booleans. Independent AP/Brier/top-capacity arithmetic, finite bounded outputs,
model/scaler/source/protocol/configuration/environment hashes and full predictions
retained privately. Original static workflow/result table rendered from actual
aggregates only, with readable labels/source/limits. No UI claim for CLI-only work.

Existing CPU, thread limit1, classical execution budget20minutes excluding input
fetch; no runtime comparison or fitted-accuracy claim for synthetic demo. Public
GitHub immutable commit and native Chrome media/owner/queue verification remain
release gates; preparation never publishes. Keep incomplete work if blocked.

Primary sources: UCI Iranian Churn provider page/licence; scikit-learn1.4.2
HistGradientBoostingClassifier official docs (native NaN routing), AP/Brier and
GroupShuffleSplit docs. Established methods; no recent-paper/SOTA claim.
