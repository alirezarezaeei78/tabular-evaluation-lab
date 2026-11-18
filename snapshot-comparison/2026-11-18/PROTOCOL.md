# Frozen-segment snapshot comparison — 18 November 2026

App-build contribution: a local command compares two RFM snapshots through the
same frozen log1p/StandardScaler/KMeans k=2 model and a development-median rule.
No new clustering algorithm, refit, model-quality improvement or deployment claim.
Freeze this contract before running the new workflow. Outcomes are unknown.

Recover the clean log_standard parameters from the verified 12 November private
models.json and its development RFM. The rule assigns 1 when at least two of
R <= development median R, F >= development median F and M >= development
median M hold; otherwise 0. This rule is an interpretability baseline, not truth.
Apply both methods to identical rows. No snapshot refitting or label alignment:
the same frozen label namespace makes the changed-label share meaningful.

Use UCI Online Retail (Chen 2015, DOI 10.24432/C5BW33, CC BY 4.0), exact Excel
SHA256 43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d.
Reuse retail_rfm cleaning/aggregation from public commit
7493181a53ab176fea85a97b6eb6cdbb82de9c5e: positive purchases, cancelled invoices
excluded, duplicate lines retained, distinct-invoice frequency, fractional recency
days. Gross positive-purchase pounds exclude returns; not net revenue.

Compare full 180-day and full 30-day RFM windows ending 2011-12-01. These are
different observation windows at one cutoff, not elapsed-time customer migration.
Development [2010-12-01,2011-07-01), 2960 customers. The benchmark and many
customers/windows have already been exposed on 5/12 November. Exploratory
functionality evidence, not independent validation, customer-type accuracy or
causal business evidence. Publish only aggregate provider results/hashes; keep
provider rows, IDs, predictions and fitted parameters private.

Report left/right/union/intersection/left-only/right-only counts, intersection
coverage on each side, ARI and changed-fixed-label count/share on matched IDs
only, complete k=2 transition cells and per-side raw RFM profiles/supports.
ARI null for fewer than two matched records; changed share null for no matches.
Suppress provider profile medians for supports <10. Retain all outcomes.

The core runner accepts bounded CSV + finite JSON parameters, never pickle, runs
locally, and writes summary JSON, assignment CSV, transition CSV and profile CSV.
Fail on malformed/duplicate IDs, schema/nonfinite/invalid values, invalid model,
oversized input or existing output directory. Empty/header-only and disjoint
cohorts must retain accounting without fabricated metrics. Independently check
ARI by exact-rational contingency, direct distance/median-rule predictions,
matched join, row-reordering invariance, unchanged input hashes and CSV/JSON
parity. Exercise real CLI, errors, empty/disjoint/single matched records.

Public default demo: authored synthetic development and two snapshots, with a
locally fitted deterministic model (seed13, n_init10) and explicit synthetic
provenance. One compact CPU fit only for the demo; no benchmark refit. Recovered
provider model must match the 12 November reference predictions. Preserve code,
actual stdout/results/configuration/dependencies/hashes. No latency claim.
Use one original static workflow/table infographic generated from actual output.
No LinkedIn publication or scheduling submission during this preparation run.
