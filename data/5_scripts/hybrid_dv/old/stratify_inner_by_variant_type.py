#!/usr/bin/env python3
"""
stratify_inner_by_variant_type.py

Analiza el dataset INNER (1↔1 loci) usando el meta CSV generado por build_join_datasets.py
y los NPZ originales (los que salen de export_dv_features.py para Illumina/ONT).

Qué reporta:
- Conteos por variant_type (raw) en INNER
- Conteos por variant_type solo para variantes (label>0)
- Mismatches de variant_type entre modalidades (vt_ill != vt_ont) dentro de INNER
- Métricas por variant_type:
    - agreement(pred): argmax(ill_probs) == argmax(ont_probs)
    - accI: argmax(ill_probs) == label
    - accO: argmax(ont_probs) == label
    - distribución de labels [0,1,2]

Cómo se lanza (desde la raíz del repo):
  python3 data/5_scripts/stratify_inner_by_variant_type.py \
    --inner_meta_csv "data/4_out/datasets/HG003/join_inner_1to1/hg003_chr20_inner_meta.csv" \
    --need_teacher 1 \
    --progress_every 5000

Argumentos:
- --inner_meta_csv: CSV de pares 1↔1 (locus, label_ill, label_ont, ill_npz, ill_idx, ont_npz, ont_idx)
- --need_teacher: 1 = usa teacher_probs/logits para predicciones; 0 = solo conteos
- --progress_every: prints de progreso cada N filas
- --dump_counts_only: 1 = solo imprime conteos (no calcula preds/accuracy)
"""

import argparse
import csv
from collections import Counter, defaultdict
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax estable para logits (N,3) -> probs (N,3)."""
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def load_npz_arrays(path: str, need_teacher: bool = True) -> dict:
    """
    Carga arrays necesarios de un NPZ original (export_dv_features output).

    Devuelve:
      - label: (N,) int64
      - variant_type: (N,) int64  (raw coding; típicamente 1=SNP, 2=INDEL)
      - probs: (N,3) float32 si need_teacher
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
    ap.add_argument("--inner_meta_csv", required=True)
    ap.add_argument("--need_teacher", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=5000)
    ap.add_argument("--dump_counts_only", type=int, default=0)
    args = ap.parse_args()

    # 1) Lee el CSV de pares INNER
    rows = []
    with open(args.inner_meta_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # 2) Agrupa por (ill_npz, ont_npz) para no recargar shards continuamente
    groups = defaultdict(list)
    for row in rows:
        groups[(row["ill_npz"], row["ont_npz"])].append(row)

    # 3) Contadores globales
    vt_counts = Counter()
    vt_counts_var = Counter()  # label>0
    vt_pair_mismatch = 0

    # 4) Métricas por variant_type
    per_vt = defaultdict(lambda: {
        "n": 0,
        "agree": 0,
        "accI": 0,
        "accO": 0,
        "lab0": 0, "lab1": 0, "lab2": 0
    })

    processed = 0

    for (ipath, opath) in sorted(groups.keys()):
        # Carga shards originales (Illumina y ONT)
        ill = load_npz_arrays(ipath, need_teacher=bool(args.need_teacher))
        ont = load_npz_arrays(opath, need_teacher=bool(args.need_teacher))

        if ill["variant_type"] is None or ont["variant_type"] is None:
            raise RuntimeError("Missing variant_type in one of the modality NPZs (check export_dv_features).")

        for row in groups[(ipath, opath)]:
            i = int(row["ill_idx"])
            o = int(row["ont_idx"])

            # Label truth-based en cada modalidad (en INNER normalmente coinciden si filtraste disagreements)
            yI = int(ill["label"][i])
            yO = int(ont["label"][o])
            y = yI  # tomamos Illumina como referencia

            vtI = int(ill["variant_type"][i])
            vtO = int(ont["variant_type"][o])
            if vtI != vtO:
                vt_pair_mismatch += 1

            vt = vtI  # estratificamos por vt de Illumina (alternativa: exigir vtI==vtO)

            vt_counts[vt] += 1
            if y > 0:
                vt_counts_var[vt] += 1

            s = per_vt[vt]
            s["n"] += 1
            s[f"lab{y}"] += 1

            if args.dump_counts_only:
                processed += 1
                continue

            pI = ill["probs"][i]
            pO = ont["probs"][o]
            predI = int(np.argmax(pI))
            predO = int(np.argmax(pO))

            s["agree"] += int(predI == predO)
            s["accI"] += int(predI == y)
            s["accO"] += int(predO == y)

            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"Processed {processed} rows...")

    # 5) Prints finales
    print("\n=== variant_type raw counts (all candidates in INNER) ===")
    for k, v in sorted(vt_counts.items()):
        print(f"variant_type={k}: {v}")

    print("\n=== variant_type raw counts (label>0 only) ===")
    for k, v in sorted(vt_counts_var.items()):
        print(f"variant_type={k}: {v}")

    print("\nNOTE: DeepVariant suele codificar SNP vs INDEL en variant_type (raw).")
    print("Mismatches vt(ill)!=vt(ont) dentro del INNER:", vt_pair_mismatch)

    if not args.dump_counts_only:
        print("\n=== Per variant_type metrics (INNER) ===")
        for vt in sorted(per_vt.keys()):
            s = per_vt[vt]
            n = s["n"]
            agree = s["agree"] / n if n else 0
            accI = s["accI"] / n if n else 0
            accO = s["accO"] / n if n else 0
            print(
                f"vt={vt} n={n} "
                f"agree(pred)={100*agree:.2f}% "
                f"accI={100*accI:.2f}% accO={100*accO:.2f}% "
                f"labels[0,1,2]=[{s['lab0']},{s['lab1']},{s['lab2']}]"
            )


if __name__ == "__main__":
    main()