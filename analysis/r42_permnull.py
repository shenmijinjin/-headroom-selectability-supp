#!/usr/bin/env python3
"""R42-A: permutation null for the observed-query aggregate oracle.

Question: how much of the context-level oracle U_c(a_c*) comes from taking a max over a small
finite set of observed queries?

Null model: preserve each query's marginal reward distribution over the K candidates, but break
the (query -> action) correspondence by independently permuting the action order within each query.
This keeps every query's per-candidate reward multiset intact while destroying any shared
cross-query action structure, so a positive aggregate oracle under the null reflects only the
finite-sample max, not a consistent good action.

We keep the antithetic (+/-) pairing in the real data and apply the SAME permutation to both
members of a pair, so the null preserves the pairing structure.

For each context: real U_c(a_c*) vs null U_c(a_c*) over R permutations; also held-out selection
gain under the same null. Uncertainty reported per passage (the independent unit).
"""
import json, os, sys
import numpy as np
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
BASE = os.path.join(RESULTS_DIR, "e2_perquery")
TASKS = ["squad", "drop", "ropes", "pwc"]
R = 200  # null permutations per context


def load(task, w):
    d = json.load(open(f"{BASE}/{w}_{task}_s42.json"))
    return [(np.asarray(c["candidate_rewards"], float), np.asarray(c["baseline_rewards"], float))
            for c in d["contexts"] if c["n_queries"] >= 2]


def ctx_oracle(cr, br):
    dr = cr - br[None, :]            # [K, nq]
    return float(dr.mean(1).max())   # max over candidates of mean-over-queries


def perm_null_ctx_oracle(cr, br, rng):
    """Permute action order independently per query, preserving antithetic pairs (K=2*pairs)."""
    K, nq = cr.shape
    dr = cr - br[None, :]
    perm = np.empty((nq, K), dtype=int)
    npair = K // 2
    for j in range(nq):
        pp = rng.permutation(npair)          # permute pair indices
        # map: for each pair p -> pp[p], keep (+,-) order within pair
        order = []
        for p in pp:
            order += [2 * p, 2 * p + 1]
        perm[j] = np.array(order)
    out = np.empty_like(dr)
    for j in range(nq):
        out[:, j] = dr[perm[j], j]
    return float(out.mean(1).max())


def main():
    res = {}
    for w in (["qwen", "gemma"] if os.path.exists(f"{BASE}/gemma_squad_s42.json") else ["qwen"]):
        res[w] = {}
        for t in TASKS:
            ctxs = load(t, w)
            real = np.array([ctx_oracle(cr, br) for cr, br in ctxs])
            null_means = np.zeros(len(ctxs))
            for ci, (cr, br) in enumerate(ctxs):
                rng = np.random.default_rng(1000 + ci)
                null_means[ci] = np.mean([perm_null_ctx_oracle(cr, br, rng) for _ in range(R)])
            exc = real - null_means
            res[w][t] = dict(n_ctx=len(ctxs), real=float(real.mean()),
                             null=float(null_means.mean()),
                             excess=float(exc.mean()),
                             excess_ci=ci_of(exc),
                             real_ci=ci_of(real), null_ci=ci_of(null_means))
            print(w, t, "real=%.4f null=%.4f excess=%+.4f CI%s" % (real.mean(), null_means.mean(), (real-null_means).mean(), [round(x,4) for x in ci_of(real-null_means)]), flush=True)
    outdir = os.path.join(RESULTS_DIR, "perm_null")
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, "report.json"), "w"), indent=2)
    print("WROTE", os.path.join(outdir, "report.json"))


def ci_of(v, n=10000):
    v = np.asarray(v); nn = len(v); rng = np.random.default_rng(0)
    b = v[rng.integers(0, nn, (n, nn))].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


if __name__ == "__main__":
    main()
