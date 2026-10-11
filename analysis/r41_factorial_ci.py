#!/usr/bin/env python3
"""Passage-clustered bootstrap CIs for the r41 factorial cells and contrasts.

Unit of resampling: the passage (pid). All rows of a passage stay together.
Per bootstrap replicate the mean is over rows, with passage multiplicity weights;
contrast = difference of the two cell means on the SAME passage resample.
Aggregate over 3 critic inits by averaging per-init replicate means.

Vectorized: per passage we precompute (sum, count) of row gains; a resample is
then just a weighted sum over passages.
"""
import json, os
import numpy as np

R = os.environ.get("RESULTS_DIR", "results")
N_BOOT = 10000
CELLS = ["A", "B", "C", "D"]
CONTRASTS = [("A", "C"), ("B", "D"), ("A", "B"), ("C", "D")]


def load(writer):
    d = json.load(open(f"{R}/e1_factorial_passage_{writer}/report.json"))
    return d["init_cells"]


def passage_stats(cell):
    """Return (up, sums, counts): per-passage sum and row count for one cell/init."""
    vals = np.asarray(cell["sel_base_per_ctx"], float)
    pids = np.asarray(cell["pid"])
    up, inv = np.unique(pids, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    sums = np.bincount(inv, weights=vals)
    return up, sums, counts


def boot_cell(per_init, n_boot=N_BOOT, seed=0):
    """Bootstrap the row-mean of a cell, averaging over inits."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([s.sum() / c.sum() for _, s, c in per_init]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        m = 0.0
        for up, s, c in per_init:
            w = np.bincount(rng.integers(0, len(up), len(up)),
                            minlength=len(up)).astype(float)
            m += (w * s).sum() / max((w * c).sum(), 1e-12)
        boots[b] = m / len(per_init)
    return point, float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def boot_contrast(per_init1, per_init2, n_boot=N_BOOT, seed=1):
    """Bootstrap of (rowmean cell1 - rowmean cell2), shared passage resample per init."""
    rng = np.random.default_rng(seed)
    per_init = []
    for (up1, s1, c1), (up2, s2, c2) in zip(per_init1, per_init2):
        up = np.union1d(up1, up2)
        def align(upx, sx, cx):
            idx = np.searchsorted(up, upx)
            S = np.zeros(len(up)); C = np.zeros(len(up))
            S[idx] = sx; C[idx] = cx
            return S, C
        S1, C1n = align(up1, s1, c1)
        S2, C2n = align(up2, s2, c2)
        per_init.append((up, S1, C1n, S2, C2n))
    def est(w, S1, C1n, S2, C2n):
        n1 = (w * C1n).sum(); n2 = (w * C2n).sum()
        m1 = (w * S1).sum() / n1 if n1 > 0 else 0.0
        m2 = (w * S2).sum() / n2 if n2 > 0 else 0.0
        return m1 - m2
    full = np.ones(len(per_init[0][0]))
    point = float(np.mean([est(np.ones(len(up)), S1, C1n, S2, C2n)
                           for up, S1, C1n, S2, C2n in per_init]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        m = 0.0
        for up, S1, C1n, S2, C2n in per_init:
            w = np.bincount(rng.integers(0, len(up), len(up)),
                            minlength=len(up)).astype(float)
            m += est(w, S1, C1n, S2, C2n)
        boots[b] = m / len(per_init)
    return point, float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def main():
    for writer in ["qwen", "gemma"]:
        try:
            inits = load(writer)
        except FileNotFoundError:
            print(f"== {writer}: report.json not found, skipping"); continue
        print(f"== {writer} ({len(inits)} inits, passage-clustered bootstrap, {N_BOOT} reps) ==")
        cells = {c: [passage_stats(ic[c]) for ic in inits] for c in CELLS}
        for c in CELLS:
            pt, lo, hi = boot_cell(cells[c])
            print(f"  {c}: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")
        for c1, c2 in CONTRASTS:
            pt, lo, hi = boot_contrast(cells[c1], cells[c2])
            print(f"  {c1}->{c2}: {pt:+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
