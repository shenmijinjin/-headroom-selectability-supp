#!/usr/bin/env python3
"""R42-B (matched): selection gain as a function of visible-query count, paired comparison.

Fixed cohort = contexts with nq >= 3. For each context and each replicate, draw one permutation
of its queries; k=1 selects on the first query, k=2 on the first two; BOTH are evaluated on the
SAME held-out remainder (perm[2:]). This isolates the effect of one extra selection query while
holding the evaluation set fixed. Report per-k gain and the paired delta k2-k1 with a
passage-clustered bootstrap CI (passage = context; replicates are Monte Carlo, not independent).
Only DROP and ROPES have enough nq>=3 contexts; squad/pwc are too sparse and excluded.
"""
import argparse, json, os
import numpy as np
from bootstrap_utils import clustered_summary, source_sha256
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
BASE = os.path.join(RESULTS_DIR, "e2_perquery")
TASKS = ["squad", "drop", "ropes", "pwc"]
R = 20


def load(task, w):
    d = json.load(open(f"{BASE}/{w}_{task}_s42.json"))
    return [(np.asarray(c["candidate_rewards"], float), np.asarray(c["baseline_rewards"], float))
            for c in d["contexts"] if c["n_queries"] >= 3]


def gain(cr, br, sel, ev):
    dr = cr - br[None, :]
    a = dr[:, sel].mean(1).argmax()
    return float(dr[a, ev].mean())


def passage_comparison(cr, br, n_replicates=R):
    """Average matched Monte Carlo draws before doing any population inference."""
    rows = []
    for r in range(n_replicates):
        perm = np.random.default_rng(1000 + r).permutation(cr.shape[1])
        g1 = gain(cr, br, perm[:1], perm[2:])
        g2 = gain(cr, br, perm[:2], perm[2:])
        rows.append([g1, g2, g2 - g1])
    return np.asarray(rows).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=os.path.join(RESULTS_DIR, 'query_count'))
    args = parser.parse_args()
    out = {"_meta": {"schema_version": 2, "monte_carlo_replicates_per_passage": R,
                     "bootstrap_unit": "passage", "estimand": "passage_equal",
                     "comparison": "first 1 versus first 2 permuted queries; common eval remainder perm[2:]",
                     "bootstrap_replicates": 10000, "bootstrap_seed": 0,
                     "fixed_cohort": "n_queries >= 3; tasks with at least 8 passages",
                     "source_sha256": {}}}
    for w in ["qwen", "gemma"]:
        out[w] = {}
        for t in TASKS:
            ctxs = load(t, w)
            out['_meta']['source_sha256'][w + '_' + t] = source_sha256(f'{BASE}/{w}_{t}_s42.json')
            if len(ctxs) < 8:
                out[w][t] = dict(n_ctx=len(ctxs), note="too few nq>=3 contexts")
                continue
            values = np.array([passage_comparison(cr, br) for cr, br in ctxs])
            summary = clustered_summary(values, [str(i) for i in range(len(ctxs))], estimand='passage_equal')
            points, intervals = summary['point'], summary['ci']
            out[w][t] = dict(n_ctx=len(ctxs), k1_gain=points[0], k1_ci=intervals[0],
                             k2_gain=points[1], k2_ci=intervals[1], delta=points[2], delta_ci=intervals[2],
                             per_passage=[dict(passage_index=i, k1_gain=float(v[0]), k2_gain=float(v[1]),
                                               delta=float(v[2])) for i, v in enumerate(values)])
            print(w, t, "n", len(ctxs), "k1=%+.4f k2=%+.4f delta=%+.4f CI%s" %
                  (*points, [round(x, 4) for x in intervals[2]]), flush=True)
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "report.json"), "w"), indent=2, allow_nan=False)
    print("WROTE", os.path.join(outdir, "report.json"))


if __name__ == "__main__":
    main()
