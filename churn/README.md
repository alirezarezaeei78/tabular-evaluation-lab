# Churn Decision Lab

A **local-only** workbench for ML developers and analysts: train a small churn comparison, inspect probabilities, apply a review budget, and export the same scored rows shown in the browser. No account, API key, paid compute, hosted demo or remote inference service.

The new contribution is a runnable model-backed scoring workflow integrating a separate sigmoid calibration partition and three explicit decision policies. This is an established-method local experiment, not a new learning algorithm or evidence that a retention offer works.

## Feature-group refit analysis, 29 October 2026

The [executed feature-ablation study](experiments/2026-10-29/README.md) refits the
same fixed booster after four predeclared group removals. It preserves all three
known partitions, all measured outcomes and a training-only one-feature reference.
The report explicitly counts new feature-equivalent rows created by column
removal. These are exploratory predictive comparisons, not causal importance or
independent validation. Its documented aggregate inspector exports CSV/JSON
without training or downloading provider data.

## Install and run

From the repository root, using Python 3.12 (tested with 3.12.9):

```sh
python -m pip install -r churn/requirements.txt
python churn/run_benchmark.py --output churn/local-output/reproduction
python churn/server.py --report churn/local-output/reproduction/results.json --port 8766
```

Open **http://127.0.0.1:8766/**. The first benchmark run downloads the small UCI provider archive and verifies its SHA-256, then trains the fixed three-seed comparison. Network access is needed for installation and that download. Scoring afterward is local. The model stays at `churn/.local/model.joblib`; the downloaded CSV, row-level audit and user uploads are not public release assets. Load only joblib files you created or trust: pickle-based formats can execute code.

1. Click **Load authored synthetic example**, or open/paste your own appropriately licensed CSV.
2. Choose prevalence, logistic, uncalibrated boosting or boosting + sigmoid.
3. Set a hypothetical exact review budget, then **Score and compare policies**.
4. Inspect the exact-budget flag, the fixed development-cutoff flag, and the 0.5 flag separately. Export JSON or CSV. Displayed scores round to six decimals; exports retain full precision.

The example has ten newly authored synthetic records and no labels. Its values/scores demonstrate the workflow, not benchmark accuracy or actual customers. The app does not save uploads or transmit them to a remote service. A selected record is a suggestion for human inspection, never an automated offer. Inputs outside the benchmark's feature distribution can produce unreliable scores even if they pass schema checks.

## Input and output

Header spelling matters. See [examples/demo.csv](examples/demo.csv) for all eleven input features. The names normalize the provider CSV's repeated spaces. Predictor fields must contain finite, nonnegative integers. Counts/durations/tenure use the provider's aggregation, not per-day values:

| Features | Meaning / accepted domain |
|---|---|
| Call Failure, Frequency of use, Frequency of SMS, Distinct Called Numbers | Aggregated counts; broad input cap of 1 billion |
| Subscription Length | Subscription months; same broad nonnegative cap |
| Seconds of Use | Aggregated call seconds; same broad cap |
| Complains | 0 or 1 |
| Charge Amount | Ordinal 0–10; the provider prose says 0–9, but the downloaded CSV actually reaches10 |
| Age Group | Ordinal 1–5 |
| Tariff Plan | 1 pay-as-you-go, 2 contractual |
| Age | Numeric years; accepted0–120 (source values15–55) |

Optional `record_id` must be unique and contain1–64 letters, numbers, hyphens or underscores. Missing IDs receive stable input-row IDs. Optional binary `Churn` supplies descriptive observed errors for that batch; it never enters the model or selection policy. Do not submit `Status`, `Customer Value` or other extra columns. Maximum input1 MB /5,000 rows. Empty, malformed, duplicate-header, nonfinite and out-of-domain inputs are rejected with a visible error; previous exports are disabled when inputs change or scoring fails.

Output includes full-precision scores, three Boolean review flags, chosen model, primary seed, budget and development cutoff. Optional supplied outcomes add TP/FP/FN/TN and precision/recall for the exact budget. Only the first50 rows render in the table; exports contain all scored rows. JSON includes report metadata; CSV contains aligned per-record fields. Undefined precision/recall is null, never invented zero evidence.

## Checked results, 21 October 2026

Data: **Iranian Churn (2020), UCI Machine Learning Repository**, DOI [10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z), [official provider](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), **CC BY 4.0**. Provider describes9-month aggregated attributes and month12 churn with a3-month planning gap. No row-level timestamps support a temporal split here. We conservatively exclude `Status` and derived `Customer Value`; their operational derivation/availability was not verified, not proven leakage. No identifier or target is a predictor.

Exact eleven-predictor groups stay together, including any conflicting-label duplicates. There are3,150 rows and2,826 groups. Seed13 was designated primary **before fitting**:1,875 fit rows,644 calibration rows and631 evaluation rows, with101 evaluation positives. Three fixed seeds were run; all outputs remain visible. Seeds29/47 are sensitivity analyses with overlapping rows, not independent replications or confidence intervals. Settings, convergence, dependencies, hardware and fit logs are in [results.json](experiments/2026-10-21/results.json).

| Same631 evaluation rows | AP ↑ | Brier loss ↓ |
|---|---:|---:|
| Training-prevalence baseline | 0.1601 | 0.13446 |
| Scaled logistic regression | 0.6831 | 0.08340 |
| Histogram gradient boosting | 0.9243 | 0.03655 |
| Same boosting + separate sigmoid fit | 0.9243 | 0.03721 |

**The calibration hypothesis was not supported by Brier loss in any declared split.** Raw→sigmoid Brier: seed13,0.036547→0.037210; seed29,0.033346→0.034906; seed47,0.030980→0.031455. AP stayed unchanged in each split. Primary log loss slightly improved0.130406→0.130251, while it worsened on the other two seeds; retain this mixed result. Brier combines calibration and discrimination. Reliability-bin supports are saved; small bins are noisy.

![Brier losses for every seed and primary reliability curves](experiments/2026-10-21/calibration.png)

At a **0.5 cutoff**, raw boosting flags99 rows (84 TP/15 FP/17 misses), sigmoid flags102 (86 TP/16 FP/15 misses). At an **exact10% budget**, both flag64 rows (63 TP/1 FP/38 misses): precision63/64 and recall63/101. A development-fixed top10% cutoff flags66 evaluation rows (65 TP/1 FP/36 misses), showing why a fixed cutoff does not guarantee future capacity. The prevalence baseline's tied development scores flag all631 rows; the app/report exposes this failure too. Exact-capacity ties use record ID ordering; fixed cutoffs include every tie. The10% capacity is hypothetical, with ceiling rounding to64/631≈10.14%.

![Primary baseline metrics and all boosting policy counts](experiments/2026-10-21/policies.png)

The default sigmoid view demonstrates the requested integration, not a test-selected winning model. Compare it with the uncalibrated model. None of these offline errors measures treatment benefit, future revenue, business savings or production reliability. Scores trained on one historical telecom source may not transfer to another organization or time period.

## Protocol, checks and reproducibility

Read the [pre-fit protocol](PROTOCOL.md), [source/licence ledger](sources.json), [configuration](experiments/2026-10-21/configuration.json), [all aggregates](experiments/2026-10-21/results.json) and [static figure data](experiments/2026-10-21/figures-data.json). The local audit retains every seed's row mapping and predictions. Public releases contain only aggregate benchmark results, hashes, authored code/demo and original static PNG/SVG figures. Future reruns/analysis of these exposed splits must be called reproduction/exploratory reuse.

```sh
python -m unittest discover -s churn -p test_churn.py -v
python churn/render_figures.py
```

The executed suite checks metric oracles with ties, Brier/count arithmetic, group separation with conflicting-label duplicates, capacity/tie/rounding edges, label-independent scoring, malformed input and HTTP errors, origin controls and export parity. Benchmark execution separately checks train-only scaler statistics, no base refit during calibration, finite probabilities and saved-model roundtrip equality. Browser checks cover the actual trained model, JSON/CSV download parity, keyboard labels/focus, error recovery and390px mobile layout. See the saved [release checks](experiments/2026-10-21/checks.json) for exact verification scope. This is not a comprehensive accessibility, security or production audit.

The previous split auditor and educational example remain preserved in this coherent repository. Authored code, demo inputs and figures: repository MIT licence. Derived UCI aggregates retain CC BY 4.0 attribution; dependency licences remain their own. No private/customer/employer data or credentials are included.
