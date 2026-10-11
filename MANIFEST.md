# Result-correspondence manifest

Every core figure/table and the artifact + script that produced it.

| Paper item | What it shows | Source artifact(s) | Script | Split / bootstrap |
|---|---|---|---|---|
| tab:headroom (T1) | P1 oracle BoK, random, ctx-free/ctx-cond same-context | `m0a_qwen`, `m0a` pools (rows 80-479) | frozen pools | row window 80-479 |
| fig:qconsist (Fig 1) | query axis: same/context-level/held-out oracle + C(c) | `q1_query_consistency/{qwen,gemma}.json` | `r8_q1_consistency.py`, `beautify_figs.py` | 20 re-splits, context resample |
| tab:qconsist (T7) | same-query vs context-level oracle + C(c) | same | same | same |
| tab:p3 (T2) | P3 passage-disjoint selected-baseline | `g4_passage_{qwen,gemma}[_ctxfree]/report.json` | `r41_g4_passage*.py` | passage thirds, passage-clustered CI |
| tab:transition (T8) | P1/P2/P3 transition | G4 + transition aggregates | frozen | as labelled |
| tab:factorial (T3), fig:factorial (Fig 2a) | 2x2 passage-disjoint | `e1_factorial_passage_{qwen,gemma}/report.json` | `r41_factorial_passage.py` | passage halves |
| fig:landscape (Fig 2b) | off-diagonal Spearman native vs canonical | `e4_landscape`, `e4_canonical`, `fig_e4_rhos.json` | `e4_*` probes, `beautify_figs.py` | shared 16-action set |
| app:permnull | permutation null for observed-query oracle | `perm_null/report.json` | `r42_permnull.py` | 200 perms/context, passage resample |
| app:querycount | selection gain vs #visible queries | `query_count/report.json` | `r42_querycount.py` | nq>=3 cohort, passage CI |
| app:canonp3 | canonicalised-P3 diagnostic + weight displacement | `canon_p3/report.json`, `canon_p3/wdisplacement_squad.json` | `r8_canon_p3.py`, `r8_wdisplacement_v2.py` | E4 96-context subset |
| app:budgetctl | writer training-budget control (Gemma 20k/80k) | `a0_search` cells | frozen | as labelled |
| fig:qconsist (Fig 1) cross-check | recompute every panel value from `e2_perquery` and diff vs frozen summary | `e2_perquery/{writer}_{task}_s42.json` | `fig2a_recompute.py` | nq>=2 filter, 20 re-splits |
| app:factorial CI | passage-clustered 95% CIs for factorial cells and contrasts | `e1_factorial_passage_{qwen,gemma}/report.json` | `r41_factorial_ci.py` | passage-clustered bootstrap, 10000 reps |

## Frozen protocol constants

- Qwen writer: Qwen3-4B D2L stage-1 @ 20000 steps; scale s*=0.475; rank 8 over 36 layers.
- Gemma writer: Gemma-2-2B D2L stage-1 @ 20000 steps; scale s*=0.412; rank 8 over 26 layers.
- Budget-control arm: Gemma stage-1 @ 80000 steps.
- Action norm ||a||^2 = 8; K=8 antithetic pairs (K=16 for the landscape probe); greedy decoding,
  max_new_tokens=256; eager attention, TF32 off.
- Selector: 2-layer width-256 pre-norm Transformer critic; AdamW lr 3e-4 wd 0.01 clip 1.0, 20 epochs,
  MSE + pairwise ranking; 3 init seeds {0,1,2}; dev-frozen coverage-0.5 abstention gate.

## Split and statistics definitions (final)

- **Split unit**: passage (rows sharing a tokenized context). Train/dev/test passage counts
  377/377/378, zero intersection asserted.
- **Bootstrap unit**: passage; all of a passage's queries/actions/seeds are resampled together
  (passage-clustered). 95% CIs are passage-clustered bootstrap.
- **Aggregation**: "macro" is task-equal mean over the four tasks unless labelled pooled.
- **Action seeds**: Qwen train {42,43,44} / dev {45,46} / test {47,48,49}; Gemma train {42,43} /
  dev {44} / test {45,46,47}.
- **Critic init seeds**: {0,1,2}.
