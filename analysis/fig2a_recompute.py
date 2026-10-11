#!/usr/bin/env python3
"""Recompute every number plotted in fig_qconsist panel (a) directly from the
passage-disjoint per-query artifact (results/e2_perquery/{writer}_{task}_s42.json),
and cross-check against the frozen summary (results/q1_query_consistency/{writer}.json)
that the paper's prose and Table cite.

Definitions (matched to the artifact fields):
  same : per-context oracle gain when each candidate is scored on the SAME query
         that selects it; averaged over queries, then contexts. Equals the mean
         per-query same-query oracle of the search pools (Table 1 BoK column).
  ctxl : context-level aggregate oracle U_c(a_c*): mean reward over all of the
         context's queries per candidate, argmax, then the aggregate reward of
         the winner. Diagnostic only (uses all query labels).
  held : held-out selection gain, mean over 20 query re-splits. Per split, the
         context's queries are permuted (rng = 20261000 + split), split into
         halves (select half gets the extra query when odd), argmax on the
         select half, gain measured on the eval half.

Panel (b)'s C(c) values are task-level Pearson correlations of candidate reward
vectors across query pairs; they live in the frozen summary only (per-query
candidate-reward matrices are retained in e2_perquery, so C(c) is recomputed
here too as a cross-check).

Writes fig2a_data.json with everything the plotting script needs, and prints a
comparison table against the frozen summary. Exits non-zero if any cross-check
mismatch exceeds 5e-3 absolute.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr

BASE = os.environ.get("RESULTS_DIR", "results")
TASKS = ["squad", "drop", "ropes", "pwc"]
N_SPLITS = 20


def load(writer, task):
    d = json.load(open(f"{BASE}/e2_perquery/{writer}_{task}_s42.json"))
    out = []
    for c in d["contexts"]:
        cr = np.asarray(c["candidate_rewards"], float)  # [K, Q]
        br = np.asarray(c["baseline_rewards"], float)   # [Q]
        if c["n_queries"] >= 2:                          # r8 load() filter
            out.append((cr, br))
    return out


def ctx_stats(cr, br, rng):
    """Frozen r8_q1_consistency.py definition, verbatim."""
    K, nq = cr.shape
    perm = rng.permutation(nq)
    nsel = (nq + 1) // 2
    sel, ev = perm[:nsel], perm[nsel:]
    dr = cr - br[None, :]
    a_star = int(np.argmax(dr[:, sel].mean(1)))
    same = float(dr[a_star, sel].mean())
    held = float(dr[a_star, ev].mean())
    a_ctx = int(np.argmax(dr.mean(1)))
    ctx_oracle = float(dr[a_ctx].mean())
    rhos = []
    for i in range(nq):
        for j in range(i + 1, nq):
            rho = spearmanr(dr[:, i], dr[:, j]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    C = float(np.mean(rhos)) if rhos else float("nan")
    return same, held, ctx_oracle, C


def per_task(ctxs):
    same_by_split = [[] for _ in range(N_SPLITS)]
    held_by_split = [[] for _ in range(N_SPLITS)]
    ctxl_vals, c_vals = [], []
    for cr, br in ctxs:
        for s in range(N_SPLITS):
            same, held, _, _ = ctx_stats(cr, br, np.random.default_rng(1000 + s))
            same_by_split[s].append(same)
            held_by_split[s].append(held)
        _, _, ctx_oracle, C = ctx_stats(cr, br, np.random.default_rng(7))
        ctxl_vals.append(ctx_oracle)
        c_vals.append(C)
    return {
        "n_ctx": len(ctxs),
        "same": float(np.mean([np.mean(v) for v in same_by_split])),
        "ctxl": float(np.mean(ctxl_vals)),
        "held_20split": float(np.mean([np.mean(v) for v in held_by_split])),
        "C": float(np.nanmean(c_vals)),
    }


def main():
    out, worst = {}, 0.0
    for w in ["qwen", "gemma"]:
        frozen = json.load(open(f"{BASE}/q1_query_consistency/{w}.json"))["tasks"]
        out[w] = {}
        print(f"== {w} ==")
        print(f"{'task':6s} {'same':>14s} {'ctxl':>14s} {'held20':>14s} {'C':>14s}")
        for t in TASKS:
            r = per_task(load(w, t))
            f = frozen[t]
            row = {}
            for k_new, k_old in [("same", "same"), ("ctxl", "ctx_oracle"),
                                 ("held_20split", "held"), ("C", "C")]:
                fv = f.get(k_old)
                nv = r[k_new]
                if fv is not None and nv is not None:
                    diff = abs(nv - fv)
                    worst = max(worst, diff)
                    row[k_new] = {"recomputed": nv, "frozen": fv, "absdiff": diff}
                else:
                    row[k_new] = {"recomputed": nv, "frozen": fv}
            out[w][t] = {"n_ctx": r["n_ctx"], **row}
            def fmt(k):
                v = row[k]
                fvv = v.get("frozen")
                return f"{v['recomputed']:+.4f}/{fvv:+.4f}" if isinstance(fvv, float) else f"{v['recomputed']:+.4f}/--"
            print(f"{t:6s} {fmt('same'):>14s} {fmt('ctxl'):>14s} {fmt('held_20split'):>14s} {fmt('C'):>14s}")
    json.dump(out, open("/tmp/fig2a_data.json", "w"), indent=2)
    print(f"\nworst |recomputed - frozen| = {worst:.5f}")
    print("WROTE /tmp/fig2a_data.json")
    if worst > 5e-3:
        print("MISMATCH above tolerance -- do NOT replot from these values without resolving")
        sys.exit(1)


if __name__ == "__main__":
    main()
