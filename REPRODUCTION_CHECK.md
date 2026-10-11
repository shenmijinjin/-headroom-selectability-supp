# Local verification record

The corrected release was checked with Python 3.12.10, NumPy 2.2.5, SciPy 1.16.2, and Matplotlib 3.10.7. The lightweight reports were regenerated in a separate copy of `results/`:

- `r8_q1_consistency.py qwen` and `gemma`: byte-identical reports.
- `r42_querycount.py`: byte-identical corrected report, with passage-level Monte Carlo aggregation.
- `r42_permnull.py`: byte-identical paired-null report.
- `fig2a_recompute.py`: strict consistency check of query-consistency figure means/correlations, tolerance 1e-12.
- `beautify_figs.py`: explicit paths from an independent working directory; source-hash manifest and PNG previews match.
- `python -m unittest discover -s analysis -p 'test_*.py' -v`: 13 focused regressions passed.
- `python -m compileall -q analysis`: passed.

No GPU fits were run during the lightweight verification above. The small canonical-P3 and supporting model-fit outputs are not claimed to have been locally reproduced; their retained summary point estimates remain explicitly labeled. See `MANIFEST.md` for each result's scope and inference contract.

## Raw-restored P3 and factorial verification (2026-10-11)

Raw per-row P3 predictions (2 writers x context-conditioned/context-free x 3 fixed critic initializations) and raw factorial cell predictions (2 writers x 4 cells x 3 inits; 566/566 passage halves, shared draws across cells) were restored on the experiment machine and bundled here. Verification, all from the bundled files:

- `python analysis/p3_statistics.py --raw results/g4_passage_<arm>/predictions.json --n-boot 10000` for each of the four arms: task-equal-row and row-pooled points and CIs match the bundled `bootstrap_summary.json` to 1e-12 on every metric.
- `python analysis/r41_factorial_ci.py --raw results/factorial_statistics/<writer>_predictions.json` for both writers: cell and contrast points match the bundled `*_factorial_ci.json` to 1e-12; CI endpoints match to 1e-9.
- `python -m unittest discover -s analysis -p 'test_*.py' -v`: 13 focused regressions passed after the update.
- `analysis/build_paper_summary.py` regenerated `results/paper_summary.json` from the restored artifacts; every P3/factorial entry now carries `recomputed_from_raw: true` with its prediction-file SHA-256.

The pre-restoration scalar-only P3 reports were moved to `results/legacy_reports/` and are not used by any manuscript number.
