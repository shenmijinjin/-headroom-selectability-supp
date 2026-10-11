#!/usr/bin/env python3
"""Paper figures for the COLING/NAACL submission.

Figures are emitted at their *final* on-page width (ACL two-column: column
219.09pt = 3.043in, text 455.24pt = 6.323in), so \\includegraphics applies no
scaling and label point sizes survive to the page. Data is hardcoded from the
frozen artifacts.

No bar charts: every panel encodes magnitude with position (dot plots, paired
slopes, ECDFs), which keeps the comparison the prose makes -- a *difference*
between two measurement regimes -- readable as a distance rather than as two
lengths the reader has to subtract by eye.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# ---- on-page geometry (pt -> in), measured from the compiled acl.sty layout ----
COL_IN  = 219.08614 / 72.27
TEXT_IN = 455.24411 / 72.27
FULL_IN = 0.98 * TEXT_IN          # figure* slot used at width=0.98\textwidth

# ---- publication style (ACL/NAACL: Times serif to match body text) ----
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.3,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.35,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.solid_capstyle": "round",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
    "pdf.fonttype": 42,
})

# Categorical slots, validated (OKLab dE, Machado CVD sim):
#   purple/green   worst adjacent dE 21.0 deutan, 31.9 normal
#   indigo/brown   worst adjacent dE 22.8 protan, 23.9 normal
C_QWEN  = "#7b3294"   # slot 1  writer: Qwen
C_GEMMA = "#1b9e77"   # slot 2  writer: Gemma
C_NAT   = "#5e4fa2"   # slot 3  native rank-gate
C_CAN   = "#a6611a"   # slot 4  SVD-canonicalised gate
INK     = "#222222"
MUTED   = "#767676"
ZERO    = "#bbbbbb"

TASKS = ["SQuAD", "DROP", "ROPES", "PwC"]


def save_at_width(fig, path, target_in, _tries=3):
    """Save `fig` so the trimmed PDF is exactly `target_in` wide on the page.

    savefig(bbox_inches="tight") trims whitespace, so the emitted width is
    smaller than figsize and \includegraphics[width=\columnwidth] silently
    scales the figure back up -- which is what shrinks label point sizes. We
    measure the trimmed result and correct figsize until it lands.
    """
    import fitz
    for _ in range(_tries):
        fig.savefig(path)
        with fitz.open(path) as d:
            got_in = d[0].rect.width / 72.0
        if abs(got_in - target_in) < 0.004:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * target_in / got_in, h, forward=True)
    return got_in


def _despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.5, pad=1.8)


# --------------------------------------------------------------------------
# Fig. 1  the query axis, in one figure.
#   (a) three oracle regimes on one DeltaF1 scale, as a paired slope per
#       (task, writer): same-query -> context-level -> held-out-query.
#       All three are the same quantity measured under different query
#       access, so one axis is correct and the retention claim reads as a
#       distance.
#   (b) query-response consistency C(c), a correlation on its own scale,
#       as a dot plot. Separate panel, never a second y-axis: aligning a
#       correlation against a DeltaF1 scale would manufacture the coupling
#       that Sec. 5.1 argues for on independent evidence.
# --------------------------------------------------------------------------
def fig_qconsist():
    # Provenance (reconciled 2026-10-10):
    #   same, ctxl -> R8 q1_query_consistency/{qwen,gemma}.json; match tab:qconsist.
    #   held       -> same artifact, mean over the 20 query re-splits. This is the
    #                 20-split protocol, NOT the canonical single split (rng=0)
    #                 whose per-task values (+0.009/-0.007/+0.022/-0.012 Qwen) and
    #                 bootstrap CIs are reported in app:heldout. Both protocols are
    #                 given there; do not mix their point estimates with each
    #                 other's CIs.
    same = {"Qwen":  [0.0554, 0.0372, 0.0480, 0.0575],
            "Gemma": [0.0961, 0.0940, 0.1080, 0.0600]}
    ctxl = {"Qwen":  [0.050, 0.030, 0.046, 0.043],
            "Gemma": [0.090, 0.088, 0.101, 0.046]}
    held = {"Qwen":  [-0.0024, -0.0096, 0.0146, -0.0072],   # 20-split means
            "Gemma": [-0.0375, -0.0015, 0.0074, -0.0023]}
    consist = {"Qwen":  [0.02, -0.01, 0.21, 0.03],
               "Gemma": [0.02, 0.08, 0.26, 0.06]}

    fig, axes = plt.subplots(2, 1, figsize=(COL_IN, 2.95), sharex=True,
                             gridspec_kw={"height_ratios": [1.62, 1],
                                          "hspace": 0.13})
    x = np.arange(len(TASKS))
    off = 0.165

    # ---- (a) three regimes, paired slopes
    ax = axes[0]
    for writer, color, dx in (("Qwen", C_QWEN, -off), ("Gemma", C_GEMMA, +off)):
        for i in range(len(TASKS)):
            xi = x[i] + dx
            ax.plot([xi, xi], [held[writer][i], same[writer][i]], color=color,
                    lw=1.5, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot([xi - 0.066, xi + 0.066], [ctxl[writer][i]] * 2, color=color,
                    lw=1.5, zorder=3)
        ax.plot(x + dx, same[writer], "o", ms=4.4, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.plot(x + dx, held[writer], "o", ms=4.4, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.2, zorder=5)
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)
    ax.set_ylabel(r"oracle gain  $\Delta$F1", labelpad=2)
    ax.set_ylim(-0.052, 0.137)
    ax.set_yticks([-0.04, 0, 0.04, 0.08, 0.12])
    _despine(ax, bottom=False)
    ax.tick_params(axis="x", length=0)

    handles = [
        Line2D([], [], color=INK, lw=0, marker="o", ms=4.4,
               markeredgecolor="white", markeredgewidth=0.8, label="same-query"),
        Line2D([], [], color=INK, lw=1.5, label="context-level"),
        Line2D([], [], color=INK, lw=0, marker="o", ms=4.4, markerfacecolor="white",
               markeredgecolor=INK, markeredgewidth=1.2, label="held-out"),
    ]
    ax.legend(handles=handles, ncol=3, loc="upper center",
              bbox_to_anchor=(0.46, 1.10), columnspacing=0.8, handlelength=1.1)

    # ---- (b) query consistency, dot plot
    ax = axes[1]
    for writer, color, dx in (("Qwen", C_QWEN, -off), ("Gemma", C_GEMMA, +off)):
        for i in range(len(TASKS)):
            ax.plot([x[i] + dx] * 2, [0, consist[writer][i]], color=color,
                    lw=0.7, alpha=0.4, zorder=2)
        ax.plot(x + dx, consist[writer], "o", ms=4.4, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=writer)
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)
    ax.set_ylabel(r"query consistency $C(c)$", labelpad=2)
    ax.set_ylim(-0.062, 0.375)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_xlim(-0.5, len(TASKS) - 0.5)
    _despine(ax)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(-0.01, 1.08))
    ax.annotate(r"$C(c)>0$ on both writers",
                xy=(2 + off + 0.07, 0.262), xytext=(3.47, 0.315),
                color=MUTED, fontsize=6.5, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                shrinkA=1, shrinkB=2))

    fig.align_ylabels(axes)
    save_at_width(fig, "fig_qconsist.pdf", COL_IN)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 2  the two evaluation gaps.
# P1 is an oracle *range* over writers x tasks (tab:transition reports
# +0.04..+0.10), so it is drawn as a span rather than a single point: pinning
# it to one writer-task value while P2/P3 are macro would mix scopes in
# exactly the way the section warns against.
# --------------------------------------------------------------------------
def fig_transition():
    p1_lo, p1_hi = 0.04, 0.10
    p1_mid = (p1_lo + p1_hi) / 2
    p2, p3 = 0.010, 0.0
    fig, ax = plt.subplots(figsize=(COL_IN, 1.78))
    x = [0, 1, 2]

    # P1 is a range over writers x tasks; P2/P3 are macro point estimates.
    ax.plot([x[0], x[0]], [p1_lo, p1_hi], color=C_NAT, lw=4.5, alpha=0.28,
            solid_capstyle="butt", zorder=2)
    ax.plot([x[0] - 0.085, x[0] + 0.085], [p1_mid] * 2, color=C_NAT, lw=1.6, zorder=3)
    ax.plot(x[1:], [p2, p3], "o", ms=5, color=C_NAT,
            markeredgecolor="white", markeredgewidth=0.9, zorder=4)
    ax.axhline(0, color=ZERO, lw=0.6, zorder=0)

    # the two gaps are the only connectors: one arrow per gap, no duplicate line
    ax.annotate("", xy=(x[1] - 0.10, p2 + 0.003), xytext=(x[0] + 0.12, p1_mid - 0.004),
                arrowprops=dict(arrowstyle="-|>", color=C_CAN, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(0.46, 0.052, "Gap 1\nquery", color=C_CAN, ha="center", va="bottom",
            fontsize=7, linespacing=1.15)
    ax.annotate("", xy=(x[2] - 0.06, p3 + 0.0012), xytext=(x[1] + 0.10, p2 - 0.0012),
                arrowprops=dict(arrowstyle="-|>", color=C_CAN, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(1.52, 0.0135, "Gap 2\ncontext", color=C_CAN, ha="center", va="bottom",
            fontsize=7, linespacing=1.15)

    ax.text(x[0], p1_hi + 0.005, "range over\nwriters $\\times$ tasks", color=MUTED,
            fontsize=6.5, ha="center", va="bottom", linespacing=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(["P1\noracle", "P2\nsame-context", "P3\ndeployment"])
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylabel(r"gain over writer  $\Delta$F1", labelpad=2)
    ax.set_ylim(-0.005, 0.132)
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    _despine(ax)
    save_at_width(fig, "regime_transition.pdf", COL_IN)
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 3  (a) factorial decomposition  (b) shared action-response ranking.
#   (a) Cleveland dot plot with the four cells on the category axis. The
#       cells are a 2x2 factorial, not a sequence, so they are never joined
#       by a line; the seen/novel-context split is drawn as a rule instead.
#   (b) ECDF rather than a histogram: it needs no bin choice, and the
#       "fraction of pairs above 0.3" the text quotes is readable off the
#       curve directly.
# --------------------------------------------------------------------------
def fig_combined():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_IN, 2.05),
                             gridspec_kw={"wspace": 0.30})

    # ---- (a) factorial cells, dot plot
    d = json.load(open("fig_e1_data.json"))
    cells = ["A", "B", "C", "D"]
    rowlab = ["A  seen ctx, seen act", "B  seen ctx, new act",
              "C  new ctx, seen act", "D  new ctx, new act"]
    y = np.arange(4)[::-1]          # A at top
    ax = axes[0]
    for i, c in enumerate(cells):
        ax.plot([d["Qwen"][c][1], d["Gemma"][c][1]], [y[i]] * 2,
                color=MUTED, lw=0.6, alpha=0.6, zorder=2)
    ax.plot([d["Qwen"][c][1] for c in cells], y, "o", ms=4.8, color=C_QWEN,
            markeredgecolor="white", markeredgewidth=0.8, label="Qwen", zorder=3)
    ax.plot([d["Gemma"][c][1] for c in cells], y, "o", ms=4.8, color=C_GEMMA,
            markeredgecolor="white", markeredgewidth=0.8, label="Gemma", zorder=3)
    ax.axvline(0, color=ZERO, lw=0.6, zorder=0)
    ax.axhline(1.5, color="#dddddd", lw=0.6, zorder=0)
    ax.text(0.080, 1.62, "seen contexts", color=MUTED, fontsize=6.5,
            ha="right", va="bottom")
    ax.text(0.080, 1.38, "novel contexts", color=MUTED, fontsize=6.5,
            ha="right", va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(rowlab, fontsize=7)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel(r"selected$-$baseline  $\Delta$F1", labelpad=2)
    ax.set_xlim(-0.028, 0.086)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax.set_title("(a) factorial decomposition", pad=4)
    _despine(ax, left=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, -0.03))

    # ---- (b) off-diagonal Spearman, native vs canonicalised, as ECDFs
    ax = axes[1]
    dd = json.load(open("fig_e4_rhos.json"))
    for key, color, lab in (("native", C_NAT, "native"),
                            ("canon", C_CAN, "canonicalised")):
        v = np.sort(np.asarray(dd[key], dtype=float))
        f = np.arange(1, v.size + 1) / v.size
        ax.step(np.concatenate(([-1.0], v)), np.concatenate(([0.0], f)),
                where="post", color=color, lw=1.3, zorder=3,
                label=r"%s (mean $%.2f$)" % (lab, v.mean()))
        ax.plot([v.mean()], [np.searchsorted(v, v.mean()) / v.size], "o", ms=4.2,
                color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax.axvline(0, color=ZERO, lw=0.6, zorder=0)
    ax.axvline(0.3, color="#dddddd", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.text(0.315, 0.06, r"$\rho>0.3$", color=MUTED, fontsize=6.5, ha="left")
    ax.set_xlabel(r"cross-context Spearman $\rho$", labelpad=2)
    ax.set_ylabel("cumulative fraction of context pairs", labelpad=2, fontsize=7.2)
    ax.set_xlim(-0.85, 1.02)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(b) shared action-response ranking", pad=4)
    _despine(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.03))

    save_at_width(fig, "fig_e1e4_combined.pdf", FULL_IN)
    plt.close(fig)


if __name__ == "__main__":
    fig_qconsist()
    fig_transition()
    fig_combined()
    print("wrote fig_qconsist.pdf, regime_transition.pdf, fig_e1e4_combined.pdf")
