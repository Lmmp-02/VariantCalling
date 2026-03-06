#!/usr/bin/env python3
import argparse, glob, csv
import numpy as np

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def kl(p, q, eps=1e-8):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob to INNER dataset npz shards")
    ap.add_argument("--out_csv", default="teacher_disagreement_top.csv")
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--use_logits_if_missing_probs", type=int, default=1)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    loci_all = []
    y_all = []
    pI_all = []
    pO_all = []
    predI_all = []
    predO_all = []

    for p in paths:
        d = np.load(p, allow_pickle=True)

        locus = d["locus"].astype(str)
        y = d["label"].astype(np.int64)

        # probs preferred
        if "ill_probs" in d.files and "ont_probs" in d.files:
            pI = d["ill_probs"].astype(np.float32)
            pO = d["ont_probs"].astype(np.float32)
        else:
            if not args.use_logits_if_missing_probs:
                raise SystemExit(f"Missing probs in {p}. Set --use_logits_if_missing_probs 1 or regenerate with include_teacher.")
            pI = softmax(d["ill_logits"].astype(np.float32), axis=1)
            pO = softmax(d["ont_logits"].astype(np.float32), axis=1)

        predI = np.argmax(pI, axis=1)
        predO = np.argmax(pO, axis=1)

        loci_all.append(locus)
        y_all.append(y)
        pI_all.append(pI)
        pO_all.append(pO)
        predI_all.append(predI)
        predO_all.append(predO)

        d.close()

    locus = np.concatenate(loci_all)
    y = np.concatenate(y_all)
    pI = np.vstack(pI_all)
    pO = np.vstack(pO_all)
    predI = np.concatenate(predI_all)
    predO = np.concatenate(predO_all)

    N = len(y)

    agree_pred = np.mean(predI == predO)
    acc_I = np.mean(predI == y)
    acc_O = np.mean(predO == y)

    both_correct = np.mean((predI == y) & (predO == y))
    I_correct_O_wrong = np.mean((predI == y) & (predO != y))
    O_correct_I_wrong = np.mean((predO == y) & (predI != y))
    both_wrong = np.mean((predI != y) & (predO != y))

    # disagreement metrics
    l1 = np.sum(np.abs(pI - pO), axis=1)
    kl_sym = kl(pI, pO) + kl(pO, pI)

    print("\n=== Teacher agreement on INNER (1↔1) ===")
    print("N:", N)
    print("Agreement argmax(pred):", f"{100*agree_pred:.2f}%")
    print("Acc Illumina vs label :", f"{100*acc_I:.2f}%")
    print("Acc ONT vs label      :", f"{100*acc_O:.2f}%")
    print("Both correct          :", f"{100*both_correct:.2f}%")
    print("Ill correct, ONT wrong:", f"{100*I_correct_O_wrong:.2f}%")
    print("ONT correct, Ill wrong:", f"{100*O_correct_I_wrong:.2f}%")
    print("Both wrong            :", f"{100*both_wrong:.2f}%")
    print("Mean L1(prob diff)    :", float(np.mean(l1)))
    print("Mean KL_sym           :", float(np.mean(kl_sym)))

    # export top_k most discrepant (by L1) + include if argmax differs
    idx = np.argsort(-l1)
    top = idx[: args.top_k]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "locus","label",
            "pred_ill","pred_ont","pred_agree",
            "pIll_0","pIll_1","pIll_2",
            "pOnt_0","pOnt_1","pOnt_2",
            "l1_prob_diff","kl_sym"
        ])
        for i in top:
            w.writerow([
                locus[i], int(y[i]),
                int(predI[i]), int(predO[i]), int(predI[i] == predO[i]),
                float(pI[i,0]), float(pI[i,1]), float(pI[i,2]),
                float(pO[i,0]), float(pO[i,1]), float(pO[i,2]),
                float(l1[i]), float(kl_sym[i])
            ])

    print("\nWrote:", args.out_csv)
    # also report how many argmax disagreements exist
    n_disagree = int(np.sum(predI != predO))
    print("Argmax disagreements:", n_disagree, f"({100*n_disagree/N:.2f}%)")

if __name__ == "__main__":
    main()