#!/usr/bin/env python3
"""R42-A: permutation null for the observed-query aggregate oracle.

Question: how much of the context-level oracle U_c(a_c*) comes from taking a max over a small
finite set of observed queries?

Null model: preserve each query's marginal reward distribution over the K candidates, but break
the (query -> action) correspondence by independently permuting the action order within each query.
This preserves each query's reward multiset. Permuting pair blocks breaks shared
pair identity while retaining within-pair orientation. The excess estimates the
observed aggregate maximum above this constrained null; it does not prove the
existence or absence of an action that generalizes to unobserved queries.

We keep the antithetic (+/-) pairing in the real data and apply the SAME permutation to both
members of a pair, so the null preserves the pairing structure.

For each context: real U_c(a_c*) vs its mean under R null permutations.
Uncertainty resamples passages and is conditional on the fixed null draws.
"""
import argparse, json, os
import numpy as np
from bootstrap_utils import clustered_summary, source_sha256
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=os.path.join(RESULTS_DIR, 'perm_null'))
    args = parser.parse_args()
    res = {'_meta': {'schema_version': 2, 'null_permutations_per_passage': R,
                     'bootstrap_unit': 'passage', 'estimand': 'passage_equal',
                     'bootstrap_replicates': 10000, 'bootstrap_seed': 0,
                     'null': 'independent permutation of adjacent antithetic pair indices per query; pair orientation retained',
                     'source_sha256': {}}}
    for w in (["qwen", "gemma"] if os.path.exists(f"{BASE}/gemma_squad_s42.json") else ["qwen"]):
        res[w] = {}
        for t in TASKS:
            ctxs = load(t, w)
            res['_meta']['source_sha256'][w + '_' + t] = source_sha256(f'{BASE}/{w}_{t}_s42.json')
            real = np.array([ctx_oracle(cr, br) for cr, br in ctxs])
            null_means = np.zeros(len(ctxs))
            for ci, (cr, br) in enumerate(ctxs):
                rng = np.random.default_rng(1000 + ci)
                null_means[ci] = np.mean([perm_null_ctx_oracle(cr, br, rng) for _ in range(R)])
            exc = real - null_means
            values = np.column_stack([real, null_means, exc])
            summary = clustered_summary(values, [str(i) for i in range(len(ctxs))])
            pt, ci = summary['point'], summary['ci']
            res[w][t] = dict(n_ctx=len(ctxs), real=pt[0], null=pt[1], excess=pt[2],
                             real_ci=ci[0], null_ci=ci[1], excess_ci=ci[2],
                             per_passage=[dict(passage_index=i, real=float(v[0]), null=float(v[1]), excess=float(v[2]))
                                          for i, v in enumerate(values)])
            print(w, t, "real=%.4f null=%.4f excess=%+.4f CI%s" % (*pt, [round(x,4) for x in ci[2]]), flush=True)
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, "report.json"), "w"), indent=2, allow_nan=False)
    print("WROTE", os.path.join(outdir, "report.json"))


def ci_of(v, n=10000):
    return clustered_summary(v, [str(i) for i in range(len(v))], n_boot=n)['ci']


if __name__ == "__main__":
    main()
