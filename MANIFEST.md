# Result and protocol manifest

This manifest maps LaTeX labels to artifacts, rather than assuming that table numbers remain fixed. Verification status refers to this package, not to an assertion that missing experiments were never performed.

| Paper item | Artifact and analysis | Verification status / estimand |
|---|---|---|
| `fig:intro` | User-supplied conceptual diagram, adapted from its editable presentation | Illustration only; no measured surface values or additional empirical inference. Paper source archive includes the figure, separately from this supplement. |
| `fig:qconsist`, `tab:qconsist`, `app:heldout` | `e2_perquery/*.json` → `r8_q1_consistency.py` → `q1_query_consistency/{qwen,gemma}.json`; check `fig2a_recompute.py` | Raw rewards bundled. Fixed 20 query partitions; passage means within task, four tasks equally weighted. Split SD is partition sensitivity, not a sampling CI. Canonical single partition is separately labeled. |
| Main selection diagnostics (`fig:factorial`, historical label) and `app:permnull` | `e2_perquery/*.json` → `r42_permnull.py` → `perm_null/report.json` | Raw rewards bundled. 200 paired-candidate permutations per passage, then paired real-minus-null passage bootstrap. No equivalence claim from intervals spanning zero. |
| Main selection diagnostics and `app:querycount` | Same raw rewards → `r42_querycount.py` → `query_count/report.json` | Raw rewards bundled. Fixed nq≥3 cohort; DROP 32 and ROPES 64 passages. k=1 and k=2 share the same evaluation remainder. Average 20 matched partitions within passage before bootstrap. |
| `tab:p3`, `app:p3_pertask`, P3 rows of `tab:transition` | `g4_passage_{qwen,gemma}[_ctxfree]/predictions.json` + `bootstrap_summary.json` (restored raw per-row predictions, 2026-10-11); inference `p3_statistics.py`; repaired generators `r41_g4_passage*.py` | Raw predictions bundled; points and CIs recompute exactly. Primary estimand `task_equal_row` with passage-clustered bootstrap (10,000 draws, seed 0); the row-pooled variant is reported as a secondary, explicitly labeled estimand. Dev gate tuning keeps the legacy row-pooled objective. |
| `tab:factorial`, `app:factorial` | `factorial_statistics/{qwen,gemma}_predictions.json` + `{qwen,gemma}_factorial_ci.json` (restored raw cell predictions, 2026-10-11); generator `r41_factorial_passage.py`, inference `r41_factorial_ci.py` | Raw predictions bundled; cells and contrasts recompute exactly with shared passage-clustered draws (10,000, seed 0). Halves (566/566) differ from P3 thirds. |
| `fig:landscape` | `figures/fig_e4_rhos.json` → `beautify_figs.py` | Scalar pairwise ranks bundled; ECDF and descriptive means reproducible. Pairs share contexts and are not treated as independent bootstrap observations. Probe-generation model inputs are not bundled. |
| `app:canonp3`, canonical part of `app:coordinate` | `canon_p3/report.json`, `canon_p3/wdisplacement_squad.json`; `r8_canon_p3.py` | Retained small diagnostic: contexts split by shared row ID across tasks, actions separately split by index. Passage-disjointness is not certified. Distinct from full P3; raw predictions absent; summary provenance requires the original experiment artifacts. |
| `tab:headroom`, `tab:headroom_full`, `tab:selctl_full`, P1/P2/shuffle rows of `tab:transition` | Original `m0a_qwen`/`m0a` candidate pools and selector outputs | Pool files and all control outputs are not bundled here. Values retained as reported; no claim of independent local reconstruction. |
| `app:budgetctl`, `app:budget`, `app:structctl`, `tab:baseline` | Original training-budget/search/capacity/scoring artifacts | Reported supporting experiments; original model inputs and all scalar outputs require the experiment machine. |

`results/paper_summary.json`, when present, consolidates exact retained points, sources, and whether they are raw-recomputed or summary-derived. It must not turn missing raw predictions into fabricated uncertainty.

## Protocol definitions

- Writers are reported as Qwen3-4B and Gemma-2-2B, LoRA rank 8, 36 and 26 layers respectively. Main writer training budget: 20,000 steps; Gemma budget-control arm: 80,000. Checkpoint identities must be recovered from experiment metadata before a final reproducibility release.
- Fixed gate scales: Qwen 0.475, Gemma 0.412. Action norm is **squared** norm `||a||² = 8`, not unit norm.
- K=8 candidates: four directions plus their negations; landscape probe K=16. “Eight antithetic pairs” would incorrectly imply 16 candidates for the main experiments.
- Decode: greedy, max 256 new tokens. SQuAD/DROP/ROPES use normalized token F1; PwC uses ROUGE-L F1. Task-equal macros are cross-metric means, not a single common F1 metric.
- P3 grouping uses equality of the **stored candidate-pool context field**. Its field type and the precise tokenizer/version are not inferable from the bundled reports. The experiment machine must verify this definition and report passage IDs/split intersections.
- Reported P3 train/dev/test unique contexts: 377/377/378; action seeds Qwen `{42,43,44}/{45,46}/{47,48,49}`, Gemma `{42,43}/{44}/{45,46,47}`. All QA rows in a context remain together. The factorial diagnostic uses seen/unseen halves (566/566 unique contexts), not these thirds.
- Critic initializations `{0,1,2}` are fixed fitted models. They do not increase the number of independent passages. The gate scans dev coverages `{0.3,0.4,0.5,0.6,0.7}` and selects the largest dev net gain; achieved test coverage need not equal 0.5.

## Corrected inference contract

`bootstrap_utils.clustered_summary` keeps point and interval estimands matched. `task_equal_row` means QA-row×action-seed mean within task, followed by equal weighting of tasks. Passage-clustered, task-stratified draws retain unequal passage row weights. `task_equal_passage`, `passage_equal`, and `row_pooled` are explicitly different estimands.

P3 and factorial first align fixed-init predictions by `(task, passage_id, row, action_seed)` and average init outcomes. The same passage multiplicities then apply to all compared outputs. Factorial inference preserves seen/unseen passage strata and uses shared draws for paired cells and contrasts. No bootstrap over independent critic-init samples is performed.

Bootstrap defaults: 10,000 percentile draws, seed 0. These intervals condition on the fitted critics, candidate pools, and Monte Carlo partitions. Query-control passages are the contexts in the raw per-query files; their exact source grouping should be checked against experiment preprocessing when recovering provenance.

## Release checks

Run the bundled regression suite and all lightweight analyses, then regenerate figures. Compare hashes in `SHA256SUMS` and `figure_provenance.json`. P3 and factorial raw predictions are bundled as of 2026-10-11: re-run `p3_statistics.py` and `r41_factorial_ci.py` against the bundled `predictions.json` files to recompute every manuscript point and interval. The superseded scalar-only P3 reports under `results/legacy_reports/` are retained for audit continuity and are not inputs to any current table.
