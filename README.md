# Headroom Is Not Selectability: anonymous supplementary material

**This archive is the supplementary material for the submission, not the paper itself.** The paper
PDF is submitted separately through the review system; what is published here is everything a
reviewer needs to verify it: the analysis and plotting code, and the small result summaries needed
to check every table and figure. Every script resolves its input and
output directories from the environment variables `RESULTS_DIR`, `SCRIPTS_DIR` and `SFA_DIR`
(defaults `results/`, `scripts/`, `sfa/` relative to the working directory); see `MANIFEST.md` for
what each script consumes and produces. The bundled `results/` directory satisfies `RESULTS_DIR`
for all consistency, permutation-null and query-count analyses. Scripts that retrain a critic or
selector (the passage-disjoint P3 and factorial runs, the selector-capacity control) additionally
need the raw candidate-reward pools and the writer's mu tokens, which are referenced by path
(`SCRIPTS_DIR`/`SFA_DIR`) because of their size; their reported outputs are already bundled and the
cross-check scripts verify the paper's numbers against them directly.

All experiments use frozen candidate pools; no script here retrains a writer. Selector/critic
training is critic-level (minutes per run).

## Layout

- `analysis/` — every reported analysis:
  - `r41_g4_passage.py`, `r41_g4_passage_gemma.py` — passage-disjoint P3 (Qwen / Gemma).
  - `r41_factorial_passage.py` — passage-disjoint 2x2 factorial decomposition (both writers).
  - `r41_factorial_ci.py` — passage-clustered bootstrap CIs for the factorial cells and contrasts
    (Appendix G numbers).
  - `r8_q1_consistency.py` — held-out-query selection gain, query-response consistency C(c),
    context-level oracle, 20-split replicates.
  - `fig2a_recompute.py` — recomputes every value in the consistency figure from the per-query
    artifacts and cross-checks against the frozen summary (max abs diff reported).
  - `r8_canon_p3.py` — canonicalised-P3 matched diagnostic (native vs SVD-canonical gate).
  - `r8_sel_matrix.py` — selector capacity control (2-layer vs 4-layer).
  - `r42_permnull.py` — permutation null for the observed-query aggregate oracle.
  - `r42_querycount.py` — selection gain vs number of visible queries.
  - `beautify_figs.py` — regenerates all paper figures from the frozen artifacts.
- `MANIFEST.md` — the result-correspondence table: every figure/table mapped to its source
  artifact, script, split, and bootstrap definition, plus the frozen protocol constants.
- `results/` — the small result summaries (aggregated numbers only, no raw dataset text) needed to
  verify every core table and the two supplementary analyses:
  - `q1_query_consistency/{qwen,gemma}.json` — frozen summaries for the consistency figure/tables.
  - `perm_null/report.json` — permutation-null results (Appendix N).
  - `query_count/report.json` — query-count results (Appendix on visible queries).
  - `g4_passage_*/report.json` — passage-disjoint P3 reports (context-conditioned and
    context-free selectors, both writers).
  - `canon_p3/report.json`, `canon_p3/wdisplacement_squad.json` — canonicalised-P3 diagnostic.
  - `e2_perquery/{writer}_{task}_s42.json` — per-query candidate rewards (candidate indices and
    scalar rewards only), the input to `r8_q1_consistency.py` and `fig2a_recompute.py`.
  - The raw factorial cell dumps (`e1_factorial_passage_*/report.json`, 20-40 MB of per-row gains)
    are omitted for size; the factorial table and its CIs are reproducible from those dumps with
    `r41_factorial_ci.py`, and the reported values are also printed by the cross-check scripts.

## Reproduction

1. Set `RESULTS_DIR` to the bundled `results/` directory (this is already the default if you run
   from the repository root, where `results/` sits next to `analysis/`).
2. Run any script in `analysis/`; each prints the numbers that appear in the paper and, where a
   frozen summary exists in `results/`, the cross-check difference. The three analyses that run
   end-to-end on the bundled data alone are `fig2a_recompute.py` (consistency figure), `r42_permnull.py`
   (permutation null) and `r42_querycount.py` (visible-query count); all three reproduce their
   frozen reports exactly.
3. Figures: `python analysis/beautify_figs.py` regenerates `fig_qconsist.pdf` and
   `fig_e1e4_combined.pdf` from the frozen summaries (its data is hardcoded from those artifacts,
   so it needs no `RESULTS_DIR`).

Resource expectations: all analysis scripts are CPU-only and finish in seconds to minutes.
