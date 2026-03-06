#!/usr/bin/env python3
"""
stratify_outer_by_variant_type.py

Analiza el OUTER join (singleton-only) usando el outer_meta.csv generado por build_join_datasets.py
y los NPZ originales de Illumina/ONT.

Qué reporta:
- Conteos por group (G11 paired / G10 Ill-only / G01 ONT-only) y variant_type
- Métricas por group y variant_type:
    - accI: pred Illumina vs label (solo donde mask_ill=1)
    - accO: pred ONT vs label (solo donde mask_ont=1)
    - agree(IvsO): argmax(ill_probs) == argmax(ont_probs) (solo donde mask_ill=mask_ont=1)
  + distribución labels[0,1,2] por grupo y tipo

Cómo se lanza:
  python3 data/5_scripts/stratify_outer_by_variant_type.py \
    --outer_meta_csv "data/4_out/datasets/HG003/join_outer_singletons/hg003_chr20_outer_meta.csv" \
    --need_teacher 1 \
    --progress_every 20000

Argumentos:
- --outer_meta_csv: CSV con filas del outer join:
    locus,label,label_ill,label_ont,mask_ill,mask_ont,ill_npz,ill_idx,ont_npz,ont_idx
- --need_teacher: 1 = usa teacher_probs/logits para predicciones; 0 = solo conteos
- --progress_every: prints de progreso cada N filas
"""

import argparse
import csv
from collections import Counter, defaultdict
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def load_arrays(path: str, need_teacher: bool = True) -> dict:
    """
    Carga arrays necesarios de un NPZ original (export_dv_features output).
    """
    d = np.load(path, allow_pickle=True)
    out = {
        "label": d["label"].astype(np.int64),
        "variant_type": d["variant_type"].astype(np.int64) if "variant_type" in d.files else None,
    }
    if need_teacher:
        if "teacher_probs" in d.files:
            out["probs"] = d["teacher_probs"].astype(np.float32)
        elif "teacher_logits" in d.files:
            out["probs"] = softmax(d["teacher_logits"].astype(np.float32))
        else:
            raise RuntimeError(f"Missing teacher_probs/logits in {path}")
    d.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer_meta_csv", required=True)
    ap.add_argument("--need_teacher", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=20000)
    args = ap.parse_args()

    # 1) Lee rows de outer_meta
    rows = []
    with open(args.outer_meta_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # 2) Agrupa por par de shards para evitar recargas
    groups = defaultdict(list)
    for row in rows:
        groups[(row["ill_npz"], row["ont_npz"])].append(row)

    # 3) Stats por group (11/10/01) y variant_type
    stats = defaultdict(lambda: defaultdict(lambda: {
        "n": 0,
        "lab0": 0, "lab1": 0, "lab2": 0,
        "accI": 0, "accO": 0, "agree": 0,
        "nI": 0, "nO": 0, "nBoth": 0,
    }))
    vt_counts = defaultdict(Counter)  # vt_counts[group][variant_type] = count

    processed = 0

    for (ipath, opath), chunk_rows in sorted(groups.items()):
        has_ill = bool(ipath)
        has_ont = bool(opath)

        ill = load_arrays(ipath, need_teacher=bool(args.need_teacher)) if has_ill else None
        ont = load_arrays(opath, need_teacher=bool(args.need_teacher)) if has_ont else None

        if has_ill and ill["variant_type"] is None:
            raise RuntimeError("Illumina NPZ missing variant_type")
        if has_ont and ont["variant_type"] is None:
            raise RuntimeError("ONT NPZ missing variant_type")

        for row in chunk_rows:
            mask_ill = int(row["mask_ill"])
            mask_ont = int(row["mask_ont"])

            # label usado en el dataset outer:
            # (por cómo lo construiste) = label Illumina si existe, si no label ONT
            y = int(row["label"])

            group = 11 if (mask_ill and mask_ont) else (10 if mask_ill else 1)  # 1==01

            iidx = int(row["ill_idx"])
            oidx = int(row["ont_idx"])

            # variant_type: preferimos Illumina si existe, si no ONT
            if mask_ill:
                vt = int(ill["variant_type"][iidx])
            else:
                vt = int(ont["variant_type"][oidx])

            vt_counts[group][vt] += 1

            s = stats[group][vt]
            s["n"] += 1
            s[f"lab{y}"] += 1

            predI = None
            predO = None

            if mask_ill:
                s["nI"] += 1
                pI = ill["probs"][iidx]
                predI = int(np.argmax(pI))
                s["accI"] += int(predI == y)

            if mask_ont:
                s["nO"] += 1
                pO = ont["probs"][oidx]
                predO = int(np.argmax(pO))
                s["accO"] += int(predO == y)

            if mask_ill and mask_ont:
                s["nBoth"] += 1
                s["agree"] += int(predI == predO)

            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"Processed {processed} rows...")

    def gname(g: int) -> str:
        return {11: "G11 (paired)", 10: "G10 (Ill-only)", 1: "G01 (ONT-only)"}[g]

    print("\n=== OUTER: counts by group and variant_type (raw) ===")
    for g in [11, 10, 1]:
        print(f"\n{gname(g)}")
        for vt, n in sorted(vt_counts[g].items()):
            print(f"  variant_type={vt}: {n}")

    print("\n=== OUTER: metrics by group and variant_type ===")
    for g in [11, 10, 1]:
        print(f"\n{gname(g)}")
        for vt in sorted(stats[g].keys()):
            s = stats[g][vt]
            n = s["n"]
            if n == 0:
                continue
            accI = s["accI"] / max(1, s["nI"])
            accO = s["accO"] / max(1, s["nO"])
            agree = s["agree"] / max(1, s["nBoth"])
            print(
                f"  vt={vt} n={n} labels[0,1,2]=[{s['lab0']},{s['lab1']},{s['lab2']}] | "
                f"accI={100*accI:.2f}% (nI={s['nI']})  "
                f"accO={100*accO:.2f}% (nO={s['nO']})  "
                f"agree(IvsO)={100*agree:.2f}% (nBoth={s['nBoth']})"
            )


if __name__ == "__main__":
    main()