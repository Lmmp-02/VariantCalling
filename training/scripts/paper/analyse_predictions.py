#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline analyses for the classical paper, run on the NPZ files written by
dump_predictions.py. CPU only; no embeddings needed.

Outputs (CSV, one row per eval set x view x model or comparison):
  view_metrics.csv     accuracy, macro-P/R/F1, error count per view
  group_errors.csv     errors per availability group (1 / 10 / 11) and class
  paired_tests.csv     paired block bootstrap (1 Mb) CIs + McNemar for
                       Hybrid vs Unimodal, Hybrid vs DV, Hybrid vs ProbFusion
  seed_summary.csv     mean / sd across training seeds (if present)

Probability-level fusion baseline ("prob_fusion"): a small MLP trained on the
DeepVariant output probabilities of both technologies plus availability masks
(8 features), using the same train / val partitions as the embedding heads.
It answers: does fusion need the 2048-d embeddings, or are DV outputs enough?

    python training/scripts/paper/analyse_predictions.py \
      --pred_dir training/out/paper/predictions --out_dir training/out/paper/analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = {"illumina": ("mask_ill", "dv_ill_probs", "unimodal_illumina"),
         "ont": ("mask_ont", "dv_ont_probs", "unimodal_ont")}
HYBRIDS = ("hybrid_simple", "hybrid_groupwise")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test (binomial, p=0.5) without scipy."""
    import math
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def cm3(y, p):
    return np.bincount(y * 3 + p, minlength=9).reshape(3, 3)


def macro(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    n = cm.sum()
    return dict(n=int(n), acc=tp.sum() / n, macro_p=prec.mean(), macro_r=rec.mean(),
                macro_f1=f1.mean(), errors=int(n - tp.sum()))


def macro_f1_from_blocks(cms_sel):
    return macro(cms_sel.sum(0))["macro_f1"]


# ---------------------------------------------------------------------
# Probability-level fusion baseline
# ---------------------------------------------------------------------

def prob_features(d):
    def lp(a):
        a = np.nan_to_num(a, nan=1.0 / 3.0)
        return np.log(np.clip(a, 1e-6, 1.0))
    mi = d["mask_ill"].astype(np.float32)[:, None]
    mo = d["mask_ont"].astype(np.float32)[:, None]
    return np.concatenate([lp(d["dv_ill_probs"]) * mi, lp(d["dv_ont_probs"]) * mo, mi, mo], axis=1)


def train_prob_fusion(train_npz, val_npz, seed=0):
    from sklearn.neural_network import MLPClassifier
    tr, va = np.load(train_npz), np.load(val_npz)
    Xtr, ytr = prob_features(tr), tr["label"].astype(int)
    Xva, yva = prob_features(va), va["label"].astype(int)
    best, best_f1 = None, -1.0
    for h in [(32,), (64, 64)]:
        clf = MLPClassifier(hidden_layer_sizes=h, max_iter=50, early_stopping=False,
                            batch_size=4096, learning_rate_init=1e-3, random_state=seed)
        clf.fit(Xtr, ytr)
        f1 = macro(cm3(yva, clf.predict(Xva)))["macro_f1"]
        print(f"[ProbFusion] hidden={h} val_macro_f1={f1:.6f}")
        if f1 > best_f1:
            best, best_f1 = clf, f1
    return best, best_f1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def load_eval_sets(pred_dir):
    sets = {}
    for f in sorted(glob.glob(str(Path(pred_dir) / "*.npz"))):
        stem = Path(f).stem
        if stem.startswith("train_split_"):
            continue
        key, tag = stem.split("__", 1)
        # priority: __all (all checkpoints) > __seeds0-4 > __seed0
        rank = 2 if tag == "all" else (1 if "seeds" in tag else 0)
        if key not in sets or rank > sets[key][1]:
            sets[key] = (f, rank)
    return sets


def model_names(d):
    return [k[len("probs__"):] for k in d.files if k.startswith("probs__")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--block_bp", type=int, default=1_000_000)
    ap.add_argument("--sklearn_prob_fusion", action="store_true",
                    help="Also train the preliminary sklearn probability-fusion baseline "
                         "(superseded by the dv_probs heads trained in the pipeline).")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)

    pf = None
    tr = Path(args.pred_dir) / "train_split_train__teacher_only.npz"
    va = Path(args.pred_dir) / "train_split_val__teacher_only.npz"
    pf_val = None
    if args.sklearn_prob_fusion and tr.exists() and va.exists():
        pf, pf_val = train_prob_fusion(tr, va)

    view_rows, group_rows, test_rows = [], [], []
    for key, (f, _) in load_eval_sets(args.pred_dir).items():
        d = np.load(f)
        y = d["label"].astype(int)
        grp = d["group"].astype(int)
        blocks = pd.factorize(pd.Series(d["dataset_id"]).astype(str) + ":" +
                              pd.Series(d["chrom"]).astype(str) + ":" +
                              (d["locus_start"] // args.block_bp).astype(str))[0]
        nb = blocks.max() + 1

        preds = {"dv_illumina": np.argmax(np.nan_to_num(d["dv_ill_probs"], nan=-1), 1),
                 "dv_ont": np.argmax(np.nan_to_num(d["dv_ont_probs"], nan=-1), 1)}
        for m in model_names(d):
            preds[m] = np.argmax(np.nan_to_num(d[f"probs__{m}"], nan=-1), 1)
        if pf is not None:
            preds["prob_fusion"] = pf.predict(prob_features(d))

        for view, (mk, _, uni) in VIEWS.items():
            sel = d[mk] == 1
            names = [n for n in preds if (n.startswith(("hybrid", "prob_fusion")) or
                                          n.startswith(f"dv_{view}") or n.startswith(uni))]
            # per-block confusion matrices for bootstrap
            bcm = {}
            for n in names:
                p = preds[n]
                cm = np.zeros((nb, 3, 3), dtype=np.int64)
                np.add.at(cm, (blocks[sel], y[sel], p[sel]), 1)
                bcm[n] = cm
                row = dict(eval_set=key, view=view, model=n, **macro(cm.sum(0)))
                view_rows.append(row)
                for g in (1, 10, 11):
                    s2 = sel & (grp == g)
                    if s2.sum() == 0:
                        continue
                    c = cm3(y[s2], p[s2])
                    group_rows.append(dict(eval_set=key, view=view, model=n, group=g,
                                           n=int(s2.sum()), errors=int(s2.sum() - np.trace(c)),
                                           het_as_homref=int(c[1, 0]), homalt_as_het=int(c[2, 1]),
                                           het_as_homalt=int(c[1, 2]),
                                           support=c.sum(1).tolist(), cm=c.tolist()))

            def compare(a, b):
                if a not in bcm or b not in bcm:
                    return
                ca, cb = bcm[a], bcm[b]
                ea = (ca.sum((1, 2)) - np.trace(ca, axis1=1, axis2=2))
                eb = (cb.sum((1, 2)) - np.trace(cb, axis1=1, axis2=2))
                d_err, d_f1 = [], []
                for _ in range(args.n_boot):
                    idx = rng.integers(0, nb, nb)
                    d_err.append(ea[idx].sum() - eb[idx].sum())
                    d_f1.append(macro_f1_from_blocks(ca[idx]) - macro_f1_from_blocks(cb[idx]))
                ok_a = preds[a][sel] == y[sel]; ok_b = preds[b][sel] == y[sel]
                n01 = int((~ok_a & ok_b).sum()); n10 = int((ok_a & ~ok_b).sum())
                p_mc = mcnemar_exact_p(n01, n10)
                test_rows.append(dict(
                    eval_set=key, view=view, model_a=a, model_b=b,
                    errors_a=int(ea.sum()), errors_b=int(eb.sum()),
                    rel_error_change=(ea.sum() - eb.sum()) / max(eb.sum(), 1),
                    d_errors_ci_lo=np.percentile(d_err, 2.5), d_errors_ci_hi=np.percentile(d_err, 97.5),
                    d_macro_f1=macro(ca.sum(0))["macro_f1"] - macro(cb.sum(0))["macro_f1"],
                    d_macro_f1_ci_lo=np.percentile(d_f1, 2.5), d_macro_f1_ci_hi=np.percentile(d_f1, 97.5),
                    mcnemar_b_only_wrong=n10, mcnemar_a_only_wrong=n01, mcnemar_p=p_mc,
                    n_blocks=int(nb)))

            # Seed-0 comparisons. Embedding heads: canonical names; DV-prob heads: __dvprobs__seed0.
            P = "__dvprobs__seed0"
            dv = f"dv_{view}"
            for h in HYBRIDS:
                compare(h, uni)                 # fusion gain, representations
                compare(h, dv)
                compare(h + P, uni + P)         # fusion gain, probabilities
                compare(h + P, dv)
                compare(h, h + P)               # representations vs probabilities (fusion)
                compare(h, "prob_fusion")
            compare(uni, dv)                    # retraining effect, representations
            compare(uni + P, dv)                # retraining effect, probabilities
            compare(uni, uni + P)               # representations vs probabilities (unimodal)
            compare("hybrid_groupwise", "hybrid_simple")
            compare("hybrid_groupwise" + P, "hybrid_simple" + P)
            compare("prob_fusion", uni)
            compare("prob_fusion", dv)

    vm = pd.DataFrame(view_rows); vm.to_csv(out / "view_metrics.csv", index=False)
    ge = pd.DataFrame(group_rows); ge.to_csv(out / "group_errors.csv", index=False)
    pd.DataFrame(test_rows).to_csv(out / "paired_tests.csv", index=False)

    # Seed summary: model names like hybrid_simple__seed3; seed 0 = canonical name.
    if not vm.empty:
        vm["family"] = vm["model"].str.replace(r"__seed\d+$", "", regex=True)
        vm["seed"] = vm["model"].str.extract(r"__seed(\d+)$")[0].fillna("0").astype(int)
        multi = vm.groupby(["eval_set", "view", "family"])["seed"].nunique()
        if (multi > 1).any():
            ss = (vm.groupby(["eval_set", "view", "family"])
                    .agg(n_seeds=("seed", "nunique"), macro_f1_mean=("macro_f1", "mean"),
                         macro_f1_sd=("macro_f1", "std"), errors_mean=("errors", "mean"),
                         errors_sd=("errors", "std")).reset_index())
            ss.to_csv(out / "seed_summary.csv", index=False)

    if not ge.empty:
        ge["family"] = ge["model"].str.replace(r"__seed\d+$", "", regex=True)
        ge["seed"] = ge["model"].str.extract(r"__seed(\d+)$")[0].fillna("0").astype(int)
        gs = (ge.groupby(["eval_set", "view", "group", "family"])
                .agg(n_seeds=("seed", "nunique"), errors_mean=("errors", "mean"), errors_sd=("errors", "std"),
                     het_as_homref_mean=("het_as_homref", "mean"), homalt_as_het_mean=("homalt_as_het", "mean"))
                .reset_index())
        gs.to_csv(out / "group_seed_summary.csv", index=False)

    meta = {"prob_fusion_val_macro_f1": None if pf_val is None else float(pf_val),
            "eval_sets": {k: v[0] for k, v in load_eval_sets(args.pred_dir).items()},
            "n_boot": args.n_boot, "block_bp": args.block_bp}
    (out / "analysis_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
