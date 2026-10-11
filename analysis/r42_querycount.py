#!/usr/bin/env python3
"""R42-B (matched): selection gain as a function of visible-query count, paired comparison.

Fixed cohort = contexts with nq >= 3. For each context and each replicate, draw one permutation
of its queries; k=1 selects on the first query, k=2 on the first two; BOTH are evaluated on the
SAME held-out remainder (perm[2:]). This isolates the effect of one extra selection query while
holding the evaluation set fixed. Report per-k gain and the paired delta k2-k1 with a
passage-clustered bootstrap CI (passage = context; replicates are Monte Carlo, not independent).
Only DROP and ROPES have enough nq>=3 contexts; squad/pwc are too sparse and excluded.
"""
import json, os
import numpy as np
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


def paci(v):
    v = np.asarray(v); nn = len(v); rng = np.random.default_rng(0)
    b = v[rng.integers(0, nn, (10000, nn))].mean(1)
    return [float(x) for x in np.quantile(b, [.025, .975])]


def main():
    out = {}
    for w in ["qwen", "gemma"]:
        out[w] = {}
        for t in TASKS:
            ctxs = load(t, w)
            if len(ctxs) < 8:
                out[w][t] = dict(n_ctx=len(ctxs), note="too few nq>=3 contexts")
                continue
            per1, per2, deltas = [], [], []
            for cr, br in ctxs:
                nq = cr.shape[1]
                for r in range(R):
                    rng = np.random.default_rng(1000 + r)
                    perm = rng.permutation(nq)
                    sel1, sel2, ev = perm[:1], perm[:2], perm[2:]
                    g1 = gain(cr, br, sel1, ev)
                    g2 = gain(cr, br, sel2, ev)
                    per1.append(g1); per2.append(g2); deltas.append(g2 - g1)
            per1 = np.array(per1); per2 = np.array(per2); deltas = np.array(deltas)
            out[w][t] = dict(n_ctx=len(ctxs), k1_gain=float(per1.mean()), k1_ci=paci(per1),
                             k2_gain=float(per2.mean()), k2_ci=paci(per2),
                             delta=float(deltas.mean()), delta_ci=paci(deltas))
            print(w, t, "n", len(ctxs), "k1=%+.4f k2=%+.4f delta=%+.4f CI%s" %
                  (per1.mean(), per2.mean(), deltas.mean(), [round(x, 4) for x in paci(deltas)]), flush=True)
    outdir = os.path.join(RESULTS_DIR, "query_count")
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, "report.json"), "w"), indent=2)
    print("WROTE", os.path.join(outdir, "report.json"))


if __name__ == "__main__":
    main()
