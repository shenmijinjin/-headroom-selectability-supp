#!/usr/bin/env python3
"""Reproduce paper figures from bundled JSON, independently of the working directory.

Root checkout: python beautify_figs.py
Anonymous package: python analysis/beautify_figs.py --output-dir figures
Explicit paths: --results-dir /path/to/results --output-dir /path/to/figures

Data point estimates and intervals are read from JSON. No fitted model, source
checkpoint, TeX installation, or PyMuPDF/fitz is needed. PDFs retain fixed physical
ACL widths; no tight crop followed by an implicit font-size rescaling is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

# Keep caches writable on machines where a home-directory cache is unavailable.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sfa-plot-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

COL_IN = 7.7 / 2.54
FULL_IN = 0.98 * 16.0 / 2.54
TASKS = ("squad", "drop", "ropes", "pwc")
LABELS = {"squad": "SQuAD", "drop": "DROP", "ropes": "ROPES", "pwc": "PwC"}
WRITERS = (("qwen", "Qwen", "#0072B2", "o"),
           ("gemma", "Gemma", "#D55E00", "s"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "legend.handletextpad": 0.4, "legend.columnspacing": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": 300,
})


class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        self.sources = {}

    def read(self, relative: str):
        p = self.root / relative
        if not p.is_file():
            raise FileNotFoundError(f"Missing bundled figure data: {p}")
        raw = p.read_bytes()
        self.sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)


def save(fig, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    # Fixed canvas preserves the specified width and the font sizes at insertion.
    fig.savefig(output / f"{stem}.pdf", bbox_inches=None)
    fig.savefig(output / f"{stem}.png", bbox_inches=None, dpi=300)
    plt.close(fig)


def interval(ax, x, y, ci, *, color, marker="o", horizontal=False, hollow=False):
    point = float(x if horizontal else y)
    if ci is None:
        errors = None
    else:
        lo, hi = map(float, ci)
        if not lo <= point <= hi:
            raise ValueError(f"Point {point} lies outside its reported interval {ci}")
        errors = np.array([[point - lo], [hi - point]])
    kw = {"xerr" if horizontal else "yerr": errors}
    ax.errorbar(x, y, **kw, fmt=marker, ms=4.3, color=color,
                mfc="white" if hollow else color, mec=color, mew=0.8,
                elinewidth=0.9, capsize=2, zorder=4)


def zero(ax, horizontal=False):
    (ax.axvline if horizontal else ax.axhline)(0, color="#AAAAAA", lw=0.7, zorder=0)


def padded_limits(values, fraction=0.15):
    lo, hi = min(0, min(values)), max(0, max(values))
    pad = max((hi - lo) * fraction, 0.002)
    return lo - pad, hi + pad


def query_figure(data: Artifacts, output: Path):
    reports = {writer: data.read(f"q1_query_consistency/{writer}.json")
               for writer, _, _, _ in WRITERS}
    fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 3.25),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.23, right=0.97, top=0.84, bottom=0.09, hspace=0.40)
    ax = axes[0]
    bounds = [0]
    for writer, _, color, _ in WRITERS:
        offset = -0.22 if writer == "qwen" else 0.22
        for i, task in enumerate(TASKS):
            d = reports[writer]["tasks"][task]
            for key, marker, hollow, protocol_offset in (("same", "o", False, -0.09),
                                        ("ctx_oracle", "_", False, 0),
                                        ("held", "^", True, 0.09)):
                ci = d.get(f"{key}_passage_ci")
                if ci is None and key == "ctx_oracle":
                    ci = d.get("ctx_oracle_ci")
                interval(ax, i + offset + protocol_offset, d[key], ci, color=color,
                         marker=marker, hollow=hollow)
                bounds.extend(ci if ci is not None else [d[key]])
    zero(ax)
    ax.set_ylim(*padded_limits(bounds))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlim(-0.5, len(TASKS) - 0.5)
    ax.set_xticks(range(len(TASKS)), [LABELS[t] for t in TASKS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Gain over writer")
    ax.set_title("(a) Query access", loc="left", pad=25)
    ax.legend(handles=[
        Line2D([], [], color="#222222", marker="o", lw=0, ms=4, label="Same"),
        Line2D([], [], color="#222222", marker="_", lw=0, ms=6, label="Aggregate"),
        Line2D([], [], color="#222222", marker="^", mfc="white", lw=0, ms=4,
               label="Held out")], ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 1.015), borderaxespad=0)
    ax = axes[1]
    bounds = [0]
    for writer, label, color, marker in WRITERS:
        offset = -0.16 if writer == "qwen" else 0.16
        for i, task in enumerate(TASKS):
            d = reports[writer]["tasks"][task]
            interval(ax, i + offset, d["C"], d["C_ci"], color=color, marker=marker)
            bounds.extend(d["C_ci"])
        ax.plot([], [], marker=marker, ls="", color=color, label=label)
    zero(ax)
    ax.set_ylim(*padded_limits(bounds, 0.12))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlim(-0.5, len(TASKS) - 0.5)
    ax.set_xticks(range(len(TASKS)), [LABELS[t] for t in TASKS])
    ax.set_ylabel(r"Consistency $C(c)$")
    ax.set_title("(b) Query-response consistency", loc="left", pad=5)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(-0.02, 1.04))
    save(fig, output, "fig_qconsist")
    return {"same_and_held": "mean across 20 fixed query partitions; partition SD is not plotted as CI",
            "intervals": "same/held: resample passages after averaging fixed partitions; aggregate/C: q1 JSON intervals",
            "width_in": COL_IN, "height_in": 3.25}


def selection_figure(data: Artifacts, output: Path):
    null = data.read("perm_null/report.json")
    count = data.read("query_count/report.json")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_IN, 2.20))
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.23, top=0.79, wspace=0.42)
    ax = axes[0]
    bounds = [0]
    for writer, label, color, marker in WRITERS:
        offset = 0.12 if writer == "qwen" else -0.12
        for i, task in enumerate(TASKS):
            d = null[writer][task]
            interval(ax, d["excess"], 3-i+offset, d["excess_ci"],
                     color=color, marker=marker, horizontal=True)
            bounds.extend(d["excess_ci"])
    zero(ax, horizontal=True)
    ax.set_yticks(range(4), [LABELS[t] for t in TASKS[::-1]])
    ax.set_ylim(-0.45, 3.45)
    ax.set_xlim(*padded_limits(bounds))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Oracle excess over permutation null")
    ax.set_title("(a) Finite-query maximization", loc="left", pad=7)
    ax = axes[1]
    bounds = [0]
    for writer, label, color, marker in WRITERS:
        offset = 0.12 if writer == "qwen" else -0.12
        for i, task in enumerate(("drop", "ropes")):
            d = count[writer][task]
            interval(ax, d["delta"], 1-i+offset, d["delta_ci"], color=color,
                     marker=marker, horizontal=True)
            bounds.extend(d["delta_ci"])
        ax.plot([], [], marker=marker, ls="", color=color, label=label)
    zero(ax, horizontal=True)
    cohort_labels = [f"{LABELS[t]} (n={count['qwen'][t]['n_ctx']})" for t in ("ropes", "drop")]
    ax.set_yticks(range(2), cohort_labels)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(*padded_limits(bounds))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Held-out gain change: 2 minus 1 query")
    ax.set_title("(b) More visible queries", loc="left", pad=7)
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls="", label=label)
                        for _, label, color, marker in WRITERS],
               loc="upper center", ncol=2, bbox_to_anchor=(0.52, 0.985))
    save(fig, output, "fig_selection_diagnostics")
    return {"intervals": "paired passage bootstrap 95% intervals from JSON",
            "query_count_cohort": "same passages with at least three queries for k=1 and k=2",
            "width_in": FULL_IN, "height_in": 2.20}


def ranking_figure(data: Artifacts, output: Path):
    d = data.read("figures/fig_e4_rhos.json")
    fig, ax = plt.subplots(figsize=(COL_IN, 2.15))
    fig.subplots_adjust(left=0.21, right=0.96, bottom=0.24, top=0.73)
    stats = {}
    for key, color, label, linestyle in (("native", "#0072B2", "Native", "-"),
                                        ("canon", "#D55E00", "SVD canonicalized", "--")):
        values = np.sort(np.asarray(d[key], dtype=float))
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid Spearman sample in {key}")
        cdf = np.arange(1, len(values)+1)/len(values)
        ax.step(np.r_[-1, values], np.r_[0, cdf], where="post", color=color,
                ls=linestyle, lw=1.3, label=f"{label} (mean {values.mean():.2f})")
        stats[key] = {"n_pairs": len(values), "mean": float(values.mean()),
                      "fraction_above_0_3": float(np.mean(values > 0.3))}
    zero(ax, horizontal=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel(r"Cross-context Spearman $\rho$")
    ax.set_ylabel("Fraction of pairs")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.025, 1.015), ncol=1)
    save(fig, output, "fig_e4_rhos")
    stats.update({"width_in": COL_IN, "height_in": 2.15})
    return stats


def main():
    here = Path(__file__).resolve().parent
    defaults = here.parent / "results" if here.name == "analysis" else here / "anonymous_repo/results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=defaults)
    parser.add_argument("--output-dir", type=Path, default=here.parent / "figures" if here.name == "analysis" else here)
    parser.add_argument("--figures", nargs="+", choices=("query", "selection", "ranking"),
                        default=("query", "selection", "ranking"))
    args = parser.parse_args()
    data = Artifacts(args.results_dir.resolve())
    output = args.output_dir.resolve()
    calls = {"query": query_figure, "selection": selection_figure, "ranking": ranking_figure}
    report = {name: calls[name](data, output) for name in args.figures}
    report["source_sha256"] = data.sources
    report["notes"] = "Query-consistency figure (fig:qconsist) partition sensitivity is not a sampling CI; legacy row-factorial data are not used."
    (output / "figure_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Generated PDF and PNG:", ", ".join(args.figures))


if __name__ == "__main__":
    main()
