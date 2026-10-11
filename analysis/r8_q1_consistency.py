#!/usr/bin/env python3
"""R8 Q1: query-consistent functional headroom + query-response consistency C(c).

Reads per-(context, candidate, query) rewards from e2_perquery/{writer}_{task}_s42.json.
All statistics resample CONTEXTS (the independent unit). The 20-split procedure reports
split-replicate SD (partition sensitivity), not a bootstrap CI over independent samples.

Outputs one JSON per writer with per-task and macro results; printed summary is the record.
"""
import json, os, os, sys
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.join(os.environ.get("RESULTS_DIR", "results"), "e2_perquery")
TASKS = ["squad", "drop", "ropes", "pwc"]
R_SPLITS = 20
N_BOOT = 10000
writer = sys.argv[1] if len(sys.argv) > 1 else "qwen"


def load(task):
    d = json.load(open(f"{BASE}/{writer}_{task}_s42.json"))
    out = []
    for c in d["contexts"]:
        cr = np.asarray(c["candidate_rewards"], dtype=np.float64)  # [K, nq]
        br = np.asarray(c["baseline_rewards"], dtype=np.float64)   # [nq]
        if c["n_queries"] >= 2:
            out.append((cr, br))
    return out


def split_half(nq, rng):
    perm = rng.permutation(nq)
    nsel = (nq + 1) // 2
    return perm[:nsel], perm[nsel:]


def ctx_stats(cr, br, rng):
    """Per-context quantities under one random select/eval query split."""
    K, nq = cr.shape
    sel, ev = split_half(nq, rng)
    dr = cr - br[None, :]                       # [K, nq] delta vs baseline
    sel_r = dr[:, sel].mean(1)                  # [K]
    a_star = int(np.argmax(sel_r))
    same = float(dr[a_star, sel].mean())
    held = float(dr[a_star, ev].mean())
    # context-level oracle uses ALL queries (diagnostic upper bound)
    a_ctx = int(np.argmax(dr.mean(1)))
    ctx_oracle = float(dr[a_ctx].mean())
    # query-response consistency across the K candidates
    if nq >= 2:
        rhos, top1, signconf, npairs = [], 0, 0, 0
        for i in range(nq):
            for j in range(i + 1, nq):
                ri, rj = dr[:, i], dr[:, j]
                rho = spearmanr(ri, rj).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
                top1 += int(np.argmax(ri) == np.argmax(rj))
                si = np.sign(ri[np.abs(ri) > 1e-9]); sj = np.sign(rj[np.abs(ri) > 1e-9])
                if len(si):
                    signconf += float(np.mean(si != sj))
                npairs += 1
        C = float(np.mean(rhos)) if rhos else np.nan
        top1 /= max(npairs, 1); signconf /= max(npairs, 1)
    else:
        C = top1 = signconf = np.nan
    base = float(br.mean())
    return dict(same=same, held=held, ctx_oracle=ctx_oracle, C=C, top1=top1,
                signconf=signconf, base=base)


def boot_ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    n = len(vals)
    if n < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    b = vals[rng.integers(0, n, (N_BOOT, n))].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    rng = np.random.default_rng(12345)          # split seeds, frozen
    out = {"writer": writer, "n_splits": R_SPLITS, "tasks": {}}
    print(f"== R8 Q1 query-consistency ({writer}) ==  CI resamples contexts (n_boot={N_BOOT})")
    macro = {k: [] for k in ["same", "held", "ctx_oracle", "C", "top1", "signconf", "retention"]}
    for task in TASKS:
        ctxs = load(task)
        n_ctx = len(ctxs)
        # split-replicate: per split, mean over contexts; report mean +- sd across splits
        sames, helds = [], []
        for r in range(R_SPLITS):
            s_r = [ctx_stats(cr, br, np.random.default_rng(1000 + r)) for cr, br in ctxs]
            sames.append(np.mean([x["same"] for x in s_r]))
            helds.append(np.mean([x["held"] for x in s_r]))
        # context-level quantities are split-independent (oracle) or averaged over splits
        per_ctx = [ctx_stats(cr, br, np.random.default_rng(7)) for cr, br in ctxs]
        ctx_or = np.array([x["ctx_oracle"] for x in per_ctx])
        C = np.array([x["C"] for x in per_ctx])
        top1 = np.array([x["top1"] for x in per_ctx])
        signconf = np.array([x["signconf"] for x in per_ctx])
        base = np.array([x["base"] for x in per_ctx])
        sames = np.array(sames); helds = np.array(helds)
        ret = helds.mean() / sames.mean() if abs(sames.mean()) > 1e-9 else float("nan")
        # predictive: C(c) vs per-context held-out retention (per split then mean), and vs ctx_oracle
        per_ctx_held, per_ctx_same = [], []
        for cr, br in ctxs:
            hh, ss = [], []
            for r in range(R_SPLITS):
                st = ctx_stats(cr, br, np.random.default_rng(1000 + r))
                hh.append(st["held"]); ss.append(st["same"])
            per_ctx_held.append(np.mean(hh)); per_ctx_same.append(np.mean(ss))
        per_ctx_held = np.array(per_ctx_held); per_ctx_same = np.array(per_ctx_same)
        ok = np.isfinite(C)
        rho_C_ret = spearmanr(C[ok], (per_ctx_held / np.where(np.abs(per_ctx_same) < 1e-9, np.nan, per_ctx_same))[ok]).statistic if ok.sum() > 5 else np.nan
        rho_C_orc = spearmanr(C[ok], ctx_or[ok]).statistic if ok.sum() > 5 else np.nan
        # partial: C vs retention controlling baseline F1 and same-query oracle (rank residuals)
        def resid(y, X):
            X = np.column_stack([np.ones(len(X)), X]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        reto = (per_ctx_held / np.where(np.abs(per_ctx_same) < 1e-9, np.nan, per_ctx_same))
        m = ok & np.isfinite(reto)
        if m.sum() > 8:
            ry = resid(reto[m], np.column_stack([base[m], per_ctx_same[m]]))
            rx = resid(C[m], np.column_stack([base[m], per_ctx_same[m]]))
            rho_partial = spearmanr(rx, ry).statistic
        else:
            rho_partial = np.nan
        ci_same, ci_held, ci_orc = boot_ci([sames.mean()]), boot_ci([helds.mean()]), boot_ci(ctx_or)
        out["tasks"][task] = dict(
            n_ctx=n_ctx, same=float(sames.mean()), same_sd=float(sames.std()),
            held=float(helds.mean()), held_sd=float(helds.std()), retention=float(ret),
            ctx_oracle=float(ctx_or.mean()), ctx_oracle_ci=boot_ci(ctx_or),
            C=float(np.nanmean(C)), C_ci=boot_ci(C), top1=float(np.nanmean(top1)),
            signconf=float(np.nanmean(signconf)),
            rho_C_retention=None if not np.isfinite(rho_C_ret) else float(rho_C_ret),
            rho_C_ctxoracle=None if not np.isfinite(rho_C_orc) else float(rho_C_orc),
            rho_C_retention_partial=None if not np.isfinite(rho_partial) else float(rho_partial))
        for k, v in [("same", sames.mean()), ("held", helds.mean()), ("ctx_oracle", ctx_or.mean()),
                     ("C", np.nanmean(C)), ("top1", np.nanmean(top1)),
                     ("signconf", np.nanmean(signconf)), ("retention", ret)]:
            macro[k].append(v)
        print(f"{task:6s} n={n_ctx:3d} same={sames.mean():+.4f}±{sames.std():.3f} "
              f"held={helds.mean():+.4f}±{helds.std():.3f} ret={ret:.2f} | "
              f"CTX_ORACLE={ctx_or.mean():+.4f} CI[{ci_orc[0]:+.4f},{ci_orc[1]:+.4f}] | "
              f"C={np.nanmean(C):+.3f} top1={np.nanmean(top1):.2f} signconf={np.nanmean(signconf):.2f} | "
              f"rho(C,ret)={rho_C_ret if np.isfinite(rho_C_ret) else float('nan'):+.2f} "
              f"rho(C,orc)={rho_C_orc if np.isfinite(rho_C_orc) else float('nan'):+.2f} "
              f"partial={rho_partial if np.isfinite(rho_partial) else float('nan'):+.2f}")
    out["macro"] = {k: float(np.mean(v)) for k, v in macro.items()}
    print("MACRO " + json.dumps(out["macro"]))
    # task-level: does mean C(c) order tasks like retention?
    Cs = [out["tasks"][t]["C"] for t in TASKS]; rets = [out["tasks"][t]["retention"] for t in TASKS]
    orcs = [out["tasks"][t]["ctx_oracle"] for t in TASKS]
    out["task_level"] = dict(
        rho_C_retention=float(spearmanr(Cs, rets).statistic),
        rho_C_ctxoracle=float(spearmanr(Cs, orcs).statistic))
    print("TASK-LEVEL rho(C,retention)=%.3f rho(C,ctx_oracle)=%.3f" %
          (out["task_level"]["rho_C_retention"], out["task_level"]["rho_C_ctxoracle"]))
    OUTDIR = os.path.join(os.environ.get("RESULTS_DIR", "results"), "q1_query_consistency")
    os.makedirs(OUTDIR, exist_ok=True)
    op = os.path.join(OUTDIR, f"{writer}.json")
    json.dump(out, open(op, "w"), indent=2)
    print("WROTE", op)


if __name__ == "__main__":
    main()
