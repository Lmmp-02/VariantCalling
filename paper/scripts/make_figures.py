"""
Figures 3 and 4 of the classical paper, from results/paper/analysis/view_metrics.csv.

Figure 3: macro-F1 at 40x, representation heads. Marker = median over five training
seeds; horizontal bar = range (min-max). DeepVariant is a single value.

Figure 4 A-B: macro-F1 against coverage, representation heads. Line = median;
band = range over seeds.
Figure 4 C-D: gain of each fusion head over the unimodal head with the same input and
seed, in percentage points of macro-F1. Marker = median over seeds; bar = interquartile
range. Solid = representations, dashed = genotype probabilities.

Outputs: paper/figures/generated/fig3{a-d}_*.pdf/png and fig4{a-d}_*.pdf/png.

    python paper/scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
CSV = REPO / "results" / "paper" / "analysis" / "view_metrics.csv"
OUT = PAPER / "figures" / "generated"

LABELS = {"dv": "DeepVariant", "unimodal": "Unimodal", "hybrid_simple": "Hybrid-S", "hybrid_groupwise": "Hybrid-G"}
COLORS = {"dv": "#4D4D4D", "unimodal": "#4C78A8", "hybrid_simple": "#F58518", "hybrid_groupwise": "#54A24B"}
MARKERS = {"dv": "o", "unimodal": "s", "hybrid_simple": "D", "hybrid_groupwise": "^"}
COV = ["5x", "10x", "20x", "40x"]
COV_LABEL = {c: c.replace("x", "\u00d7") for c in COV}


def load() -> pd.DataFrame:
    v = pd.read_csv(CSV)
    v["family"] = v["model"].str.replace(r"__seed\d+$", "", regex=True)
    v["seed"] = v["model"].str.extract(r"__seed(\d+)$")[0].fillna("0").astype(int)
    v["input"] = np.where(v["family"].str.contains("__dvprobs"), "probs", "repr")
    base = v["family"].str.replace("__dvprobs", "", regex=False)
    v["hd"] = np.select(
        [base.str.startswith("dv_"), base.str.startswith("unimodal_"), base == "hybrid_simple", base == "hybrid_groupwise"],
        ["dv", "unimodal", "hybrid_simple", "hybrid_groupwise"], default="other")
    return v


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 3
def fig3_at_40x(v):
    panels = [("hg004_chr20_40x", "illumina", "fig3a_hg004_illumina", "HG004 chr20 \u2014 Illumina"),
              ("hg004_chr20_40x", "ont", "fig3b_hg004_ont", "HG004 chr20 \u2014 ONT"),
              ("hg005_40x", "illumina", "fig3c_hg005_illumina", "HG005 chr20+chr21 \u2014 Illumina"),
              ("hg005_40x", "ont", "fig3d_hg005_ont", "HG005 chr20+chr21 \u2014 ONT")]
    order = ["dv", "unimodal", "hybrid_simple", "hybrid_groupwise"]
    for es, view, stem, title in panels:
        sub = v[(v.eval_set == es) & (v.view == view) & (v.input == "repr") & (v.hd.isin(order))]
        fig, ax = plt.subplots(figsize=(5.6, 3.35))
        y = {h: len(order) - 1 - i for i, h in enumerate(order)}
        for h in order:
            vals = sub[sub.hd == h]["macro_f1"].to_numpy()
            med = float(np.median(vals))
            if len(vals) > 1:
                ax.hlines(y[h], vals.min(), vals.max(), color=COLORS[h], linewidth=2.2, alpha=0.55)
            ax.plot(med, y[h], MARKERS[h], color=COLORS[h], markeredgecolor="black", markersize=7, zorder=3)
            ax.text(max(vals.max(), med) + 0.00004, y[h], f"{med:.4f}", va="center", fontsize=8)
            if h == "dv":
                ax.axvline(med, color="#9A9A9A", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_yticks(list(y.values()))
        ax.set_yticklabels([LABELS[h] for h in y])
        ax.set_xlim(0.9965, 1.0)
        ax.set_xticks([0.997, 0.998, 0.999, 1.000])
        ax.set_xlabel("Macro-F1")
        ax.set_title(title, fontsize=10, loc="left")
        style(ax)
        save(fig, stem)


# ---------------------------------------------------------------- Figure 4
def fig4_coverage(v):
    hg5 = v[v.eval_set.str.startswith("hg005_")].copy()
    hg5["cov"] = hg5.eval_set.str.replace("hg005_", "", regex=False)
    x = {c: i for i, c in enumerate(COV)}

    for view, stem, title in [("illumina", "fig4a_coverage_illumina_absolute", "Illumina"),
                              ("ont", "fig4b_coverage_ont_absolute", "ONT")]:
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        for h in ["dv", "unimodal", "hybrid_simple", "hybrid_groupwise"]:
            sub = hg5[(hg5.view == view) & (hg5.input == "repr") & (hg5.hd == h)]
            g = sub.groupby("cov")["macro_f1"]
            med = g.median().reindex(COV); lo = g.min().reindex(COV); hi = g.max().reindex(COV)
            xs = [x[c] for c in COV]
            if h != "dv":
                ax.fill_between(xs, lo, hi, color=COLORS[h], alpha=0.15, linewidth=0)
            ax.plot(xs, med, "-", marker=MARKERS[h], color=COLORS[h], markeredgecolor="black",
                    markersize=6, label=LABELS[h])
        ax.set_xticks(range(len(COV))); ax.set_xticklabels([COV_LABEL[c] for c in COV])
        ax.set_xlabel("Coverage"); ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.75, 1.005); ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
        save(fig, stem)

    for view, stem, title in [("illumina", "fig4c_coverage_illumina_gain", "Illumina \u2014 gain over unimodal"),
                              ("ont", "fig4d_coverage_ont_gain", "ONT \u2014 gain over unimodal")]:
        fig, ax = plt.subplots(figsize=(5.8, 3.35))
        offs = {("hybrid_simple", "repr"): -0.12, ("hybrid_groupwise", "repr"): -0.04,
                ("hybrid_simple", "probs"): 0.04, ("hybrid_groupwise", "probs"): 0.12}
        for (h, inp), off in offs.items():
            meds, q1s, q3s = [], [], []
            for c in COV:
                sub = hg5[(hg5.view == view) & (hg5.input == inp) & (hg5["cov"] == c)]
                u = sub[sub.hd == "unimodal"].set_index("seed")["macro_f1"]
                hh = sub[sub.hd == h].set_index("seed")["macro_f1"]
                gain = (hh - u).dropna() * 100
                meds.append(gain.median()); q1s.append(gain.quantile(0.25)); q3s.append(gain.quantile(0.75))
            xs = np.arange(len(COV)) + off
            meds, q1s, q3s = map(np.array, (meds, q1s, q3s))
            ls = "-" if inp == "repr" else "--"
            ax.errorbar(xs, meds, yerr=[meds - q1s, q3s - meds], fmt="none", ecolor=COLORS[h], elinewidth=1.2, alpha=0.8)
            ax.plot(xs, meds, ls, marker=MARKERS[h], color=COLORS[h], markeredgecolor="black", markersize=6,
                    markerfacecolor=COLORS[h] if inp == "repr" else "white",
                    label=f"{LABELS[h]} ({'repr.' if inp == 'repr' else 'prob.'})")
        ax.axhline(0, color="#9A9A9A", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(COV))); ax.set_xticklabels([COV_LABEL[c] for c in COV])
        ax.set_xlabel("Coverage"); ax.set_ylabel("Gain over unimodal (pp)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
        save(fig, stem)


if __name__ == "__main__":
    data = load()
    fig3_at_40x(data)
    fig4_coverage(data)
    print("Saved to", OUT)
