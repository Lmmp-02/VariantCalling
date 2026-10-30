"""
LaTeX table bodies of the classical paper, from the result files in the repository.

    Table 2  tab2_errors_40x.tex         errors at 40x, median [range], representation heads
    Table 3  tab3_errors_groups.tex      errors by availability group, single / shared (medians)
    Table 4  tab4_input_types.tex        errors with representations and probabilities (medians)
    Table 6  tab6_endtoend_deepvariant.tex  end-to-end DeepVariant on HG003 chr20 (hap.py)
    Table 7  tab7_group_composition.tex  size and genotype composition of the availability groups
    Table 8  tab8_singleton_audit.tex    effect of the singleton-locus rule on HG005 at 40x
    Table 9  tab9_seeds.tex              errors per input type, median [range], all heads

Tables 1 and 5 come from the HG003 chr20 exploratory study
(studies/hg003_chr20_initial_validation/).

Each file holds the rows between \\midrule and \\bottomrule of the matching table.

    python paper/scripts/make_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
ANALYSIS = REPO / "results" / "paper" / "analysis"
AUDIT = REPO / "results" / "paper" / "singleton_audit"
ENDTOEND = REPO / "studies" / "deepvariant_low_coverage" / "results" / "benchmark_summary.csv"
OUT = PAPER / "tables"

X = r"$\times$"
EVAL_SETS = [("hg004_chr20_40x", f"HG004 40{X}"), ("hg005_40x", f"HG005 40{X}"),
             ("hg005_20x", f"HG005 20{X}"), ("hg005_10x", f"HG005 10{X}"),
             ("hg005_5x", f"HG005 5{X}")]
VIEWS = [("illumina", "Illumina"), ("ont", "ONT")]


def add_family_seed(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["family"] = d.model.str.replace(r"__seed\d+$", "", regex=True)
    d["seed"] = d.model.str.extract(r"__seed(\d+)$")[0].fillna("0").astype(int)
    return d


v = add_family_seed(pd.read_csv(ANALYSIS / "view_metrics.csv"))
g = add_family_seed(pd.read_csv(ANALYSIS / "group_errors.csv"))


def fam(view, head, inp="r"):
    """Model family name. head: dv | uni | hs | hg; inp: r (representations) | p (probabilities)."""
    if head == "dv":
        return f"dv_{view}"
    base = {"uni": f"unimodal_{view}", "hs": "hybrid_simple", "hg": "hybrid_groupwise"}[head]
    return base + ("__dvprobs" if inp == "p" else "")


def n(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x):
    return f"{100 * x:.1f}"


def med_range(vals):
    vals = np.asarray(vals)
    if len(vals) == 1:
        return n(vals[0])
    return f"{n(np.median(vals))} [{n(vals.min())}--{n(vals.max())}]"


def errs(es, view, head, inp="r"):
    sel = (v.eval_set == es) & (v.view == view) & (v.family == fam(view, head, inp))
    return v[sel].sort_values("seed").errors.to_numpy()


def table2():
    rows = []
    for es, lab in [("hg004_chr20_40x", "HG004 chr20"), ("hg005_40x", "HG005 chr20+21")]:
        for view, vl in VIEWS:
            N = int(v[(v.eval_set == es) & (v.view == view)].n.iloc[0])
            cells = [med_range(errs(es, view, h)) for h in ["dv", "uni", "hs", "hg"]]
            rows.append(f"        {lab} & {vl} & {n(N)} & " + " & ".join(cells) + r" \\")
    return rows


def table3():
    rows = []
    for es, lab in EVAL_SETS:
        for view, vl, single in [("illumina", "Illumina", 10), ("ont", "ONT", 1)]:
            sel = (g.eval_set == es) & (g.view == view)
            n_single = int(g[sel & (g.group == single)].n.iloc[0])
            n_shared = int(g[sel & (g.group == 11)].n.iloc[0])
            cells = []
            for h in ["dv", "uni", "hs", "hg"]:
                sub = g[sel & (g.family == fam(view, h))]
                a = sub[sub.group == single].errors.median()
                b = sub[sub.group == 11].errors.median()
                cells.append(f"{n(a)}\\,/\\,{n(b)}")
            rows.append(f"        {lab} & {vl} & {n(n_single)} & {n(n_shared)} & " + " & ".join(cells) + r" \\")
        if es == "hg004_chr20_40x":
            rows.append(r"        \midrule")
    return rows


def table4():
    rows = []
    for cov in ["40x", "20x", "10x", "5x"]:
        es = f"hg005_{cov}"
        for view, vl in VIEWS:
            cells = [n(np.median(errs(es, view, h, i))) for i in ["r", "p"] for h in ["uni", "hs", "hg"]]
            rows.append(f"        {cov[:-1]}{X} & {vl} & {n(errs(es, view, 'dv')[0])} & " + " & ".join(cells) + r" \\")
    return rows


def table6():
    d = pd.read_csv(ENDTOEND)
    rows = []
    for tech, tl in [("Illumina", "Illumina"), ("ONT R104", "ONT")]:
        for cov in ["30x", "15x", "10x", "5x"]:
            cells = []
            for var in ["SNP", "INDEL"]:
                r = d[(d.Tech == tech) & (d.Coverage == cov) & (d.Variant == var)].iloc[0]
                cells += [f"{r.Precision:.4f}", f"{r.Recall:.4f}", f"{r.F1:.4f}"]
            rows.append(f"        {tl} & {cov[:-1]}{X} & " + " & ".join(cells) + r" \\")
        if tech == "Illumina":
            rows.append(r"        \midrule")
    return rows


def table7():
    """Genotype composition from the class support of the DeepVariant rows (GIAB-derived labels)."""
    dv = g[g.model.isin(["dv_illumina", "dv_ont"])]
    groups = [(10, "illumina", "Illumina only (10)"), (1, "ont", "ONT only (1)"), (11, "illumina", "Both (11)")]
    rows = []
    for es, lab in [e for e in EVAL_SETS]:
        for i, (grp, view, gl) in enumerate(groups):
            r = dv[(dv.eval_set == es) & (dv.view == view) & (dv.group == grp)].iloc[0]
            sup = np.asarray(json.loads(r.support), dtype=float)
            first = lab if i == 0 else ""
            rows.append(f"        {first} & {gl} & {n(r.n)} & " + " & ".join(pct(s / sup.sum()) for s in sup) + r" \\")
        if es != EVAL_SETS[-1][0]:
            rows.append(r"        \midrule")
    return rows


def table8():
    loci = pd.read_csv(AUDIT / "singleton_loci_global_summary.csv").iloc[0]
    s = pd.read_csv(AUDIT / "singleton_loci_summary.csv")
    rows_by = {(r.dataset, int(r.kept_by_outer_singleton_rule)): int(r.rows) for r in s.itertuples()}

    def line(label, before, kept):
        return f"        {label} & {n(before)} & {n(kept)} & {100 * (before - kept) / before:.1f} \\\\"

    ill_b, ill_k = rows_by[("illumina", 0)] + rows_by[("illumina", 1)], rows_by[("illumina", 1)]
    ont_b, ont_k = rows_by[("ont", 0)] + rows_by[("ont", 1)], rows_by[("ont", 1)]
    return [line("Unique loci", int(loci.all_loci), int(loci.keep_loci)),
            line("Illumina candidate rows", ill_b, ill_k),
            line("ONT candidate rows", ont_b, ont_k),
            line("All candidate rows", ill_b + ont_b, ill_k + ont_k)]


def table9():
    rows = []
    for es, lab in EVAL_SETS:
        for view, vl in [("illumina", "Ill."), ("ont", "ONT")]:
            for inp, il in [("r", "Repr."), ("p", "Prob.")]:
                cells = [med_range(errs(es, view, h, inp)) for h in ["uni", "hs", "hg"]]
                rows.append(f"        {lab} & {vl} & {il} & " + " & ".join(cells) + r" \\")
    return rows


TABLES = {
    "tab2_errors_40x": table2,
    "tab3_errors_groups": table3,
    "tab4_input_types": table4,
    "tab6_endtoend_deepvariant": table6,
    "tab7_group_composition": table7,
    "tab8_singleton_audit": table8,
    "tab9_seeds": table9,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, fn in TABLES.items():
        (OUT / f"{stem}.tex").write_text("\n".join(fn()) + "\n", encoding="utf-8")
        print("Saved", (OUT / f"{stem}.tex").relative_to(REPO))


if __name__ == "__main__":
    main()
