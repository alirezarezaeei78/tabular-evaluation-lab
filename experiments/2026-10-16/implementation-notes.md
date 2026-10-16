# Implementation and evaluation record

The protocol, configuration, fixture generator and seeds were declared before the first experiment. The first run passed 41 correctness/CLI checks and matched the independent generator counts for all ten cases. It is retained under `initial-run/`.

During implementation, before the first ten-case experiment, the parser was strengthened to hash the same bounded byte snapshot it actually parsed, avoiding a second-read provenance mismatch if a local file changed between reads. Before final packaging, three focused checks were added for exact snapshot bytes, file-size rejection and row-count rejection. The final experiment also verifies that every original synthetic row ID occurs exactly once across each 600-row partition. These additions do not change the declared equivalence policy, partition memberships, labels or overlap results. All final fixture counts must match the initial run.

The earlier runner wrote a fixed example command in result metadata, even when a different output directory was supplied. The final runner records the actual requested directory. Earlier runs remain preserved; this metadata repair does not alter counts.

The synthetic cases and results are visible during development and remain functionality/stress testing. No threshold, split or model was tuned to improve published counts. No models were fitted, and no measured latency or independent-validation claim is made.

Browser use was disabled in Work settings during preparation. No browser workflow or GitHub/LinkedIn publication is claimed until separately verified. The tool is a local CLI/API; no hosted or browser interface is part of this release.
