#!/usr/bin/env python3
"""
build_hybrid_pairs_by_locus.py

Construye un dataset multimodal a partir de NPZs (Illumina y ONT) usando:

- Key base: locus (chr:start-end)
- Matching dentro de cada locus compartido (opcional pero recomendado): greedy 1:1 por distancia L1 en teacher_probs.
  (Evita medias de embeddings. Cada fila sigue siendo un ejemplo real.)

Genera dos modos:
  1) merged:   una fila por par (both) + filas unimodales (ill-only / ont-only)
  2) expanded: mantiene todas las filas originales (ill + ont). Si hay match, añade la otra modalidad.
              Incluye row_weight=0.5 para pares (para no duplicar su contribución al entrenar).

Salida en shards NPZ para no reventar RAM.

Uso:
  python scripts/build_hybrid_pairs_by_locus.py \
    --illumina_glob "data/4_out/datasets/HG003/Illumina/chr20/cov48x_sorig/*.npz" \
    --ont_glob "data/4_out/datasets/HG003/ONT/chr20/cov40x_sorig/*.npz" \
    --out_prefix "data/4_out/datasets/HG003/hybrid/chr20/HG003.hybrid" \
    --mode merged \
    --chunk_size 50000

Notas:
- Requiere que los NPZ tengan al menos: locus, embeddings, teacher_logits, teacher_probs.
- Si tienen variant_type/sequencing_type, se usa solo para reporting.
"""

import argparse, glob, os, re
from collections import defaultdict
import numpy as np

def parse_locus(locus: str):
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", locus)
    if not m:
        return ("", -1, -1)
    return (m.group(1), int(m.group(2)), int(m.group(3)))

def load_npz_list(paths):
    """Carga y concatena arrays (incluye embeddings)."""
    loci = []
    emb = []
    lg = []
    pb = []
    vt = []
    st = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        loci.append(d["locus"].astype(str))
        emb.append(d["embeddings"].astype(np.float32))
        lg.append(d["teacher_logits"].astype(np.float32))
        pb.append(d["teacher_probs"].astype(np.float32))
        vt.append(d["variant_type"] if "variant_type" in d.files else np.full(len(d["locus"]), -1, dtype=np.int16))
        st.append(d["sequencing_type"] if "sequencing_type" in d.files else np.full(len(d["locus"]), -1, dtype=np.int16))
    return (
        np.concatenate(loci),
        np.concatenate(emb),
        np.concatenate(lg),
        np.concatenate(pb),
        np.concatenate(vt).astype(np.int16),
        np.concatenate(st).astype(np.int16),
    )

def build_index_by_locus(locus_arr):
    idx = defaultdict(list)
    for i, loc in enumerate(locus_arr):
        idx[loc].append(i)
    return idx

def greedy_match_by_probs(ill_idx, ont_idx, probs_ill, probs_ont):
    """
    Match 1:1 dentro del locus usando greedy por menor distancia L1 en probs.
    Devuelve:
      pairs: list[(i_idx, o_idx, cost)]
      ill_unmatched: list[i_idx]
      ont_unmatched: list[o_idx]
    """
    if not ill_idx or not ont_idx:
        return [], ill_idx, ont_idx

    I = np.array(ill_idx, dtype=np.int64)
    O = np.array(ont_idx, dtype=np.int64)

    Pi = probs_ill[I]  # (k,3)
    Po = probs_ont[O]  # (m,3)

    # matriz de costes L1
    # cost[k,m] = sum(|Pi - Po|)
    cost = np.abs(Pi[:, None, :] - Po[None, :, :]).sum(axis=2)

    pairs = []
    used_i = set()
    used_o = set()

    # greedy global: repetidamente coger el mínimo restante
    # (k,m suelen ser pequeños; esto es OK para chr20)
    while True:
        # invalida filas/cols ya usadas
        c = cost.copy()
        if used_i:
            c[list(used_i), :] = np.inf
        if used_o:
            c[:, list(used_o)] = np.inf

        kmin = np.argmin(c)
        v = c.flat[kmin]
        if not np.isfinite(v):
            break

        i = kmin // c.shape[1]
        j = kmin % c.shape[1]
        used_i.add(i); used_o.add(j)
        pairs.append((int(I[i]), int(O[j]), float(v)))

    ill_unmatched = [int(I[i]) for i in range(len(I)) if i not in used_i]
    ont_unmatched = [int(O[j]) for j in range(len(O)) if j not in used_o]
    return pairs, ill_unmatched, ont_unmatched

def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def write_shard(out_path, payload: dict):
    ensure_dir(out_path)
    np.savez_compressed(out_path, **payload)
    print("Saved", out_path, "N=", len(payload["locus"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--illumina_glob", required=True)
    ap.add_argument("--ont_glob", required=True)
    ap.add_argument("--out_prefix", required=True, help="prefix sin _0000.npz")
    ap.add_argument("--mode", choices=["merged", "expanded"], default="merged")
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--no_match", action="store_true", help="No hace matching; solo marca both por locus (para debug)")
    args = ap.parse_args()

    ill_paths = sorted(glob.glob(args.illumina_glob))
    ont_paths = sorted(glob.glob(args.ont_glob))
    if not ill_paths:
        raise SystemExit(f"No Illumina NPZ matched: {args.illumina_glob}")
    if not ont_paths:
        raise SystemExit(f"No ONT NPZ matched: {args.ont_glob}")

    print("Loading Illumina...")
    ill_locus, ill_emb, ill_lg, ill_pb, ill_vt, ill_st = load_npz_list(ill_paths)
    print("Illumina N:", len(ill_locus), "D:", ill_emb.shape[1])

    print("Loading ONT...")
    ont_locus, ont_emb, ont_lg, ont_pb, ont_vt, ont_st = load_npz_list(ont_paths)
    print("ONT N:", len(ont_locus), "D:", ont_emb.shape[1])

    D = ill_emb.shape[1]
    assert ont_emb.shape[1] == D, "Embedding dims differ between modalities."

    idx_ill = build_index_by_locus(ill_locus)
    idx_ont = build_index_by_locus(ont_locus)

    loci_all = sorted(set(idx_ill.keys()) | set(idx_ont.keys()))
    loci_both = sorted(set(idx_ill.keys()) & set(idx_ont.keys()))
    print("Unique loci Ill:", len(idx_ill), "ONT:", len(idx_ont), "Union:", len(loci_all), "Both:", len(loci_both))

    # Output accumulators (sharded)
    out = {k: [] for k in [
        "locus","chrom","locus_start","locus_end",
        "mask_ill","mask_ont",
        "emb_ill","emb_ont",
        "logits_ill","logits_ont",
        "probs_ill","probs_ont",
        "src_ill","src_ont",
        "match_cost","pair_id",
        "row_weight",
    ]}

    shard = 0
    pair_counter = 0

    def flush():
        nonlocal shard
        if not out["locus"]:
            return
        payload = {
            "locus": np.asarray(out["locus"], dtype=np.str_),
            "chrom": np.asarray(out["chrom"], dtype=np.str_),
            "locus_start": np.asarray(out["locus_start"], dtype=np.int32),
            "locus_end": np.asarray(out["locus_end"], dtype=np.int32),

            "mask_ill": np.asarray(out["mask_ill"], dtype=np.int8),
            "mask_ont": np.asarray(out["mask_ont"], dtype=np.int8),

            "emb_ill": np.stack(out["emb_ill"]).astype(np.float32),
            "emb_ont": np.stack(out["emb_ont"]).astype(np.float32),

            "logits_ill": np.stack(out["logits_ill"]).astype(np.float32),
            "logits_ont": np.stack(out["logits_ont"]).astype(np.float32),

            "probs_ill": np.stack(out["probs_ill"]).astype(np.float32),
            "probs_ont": np.stack(out["probs_ont"]).astype(np.float32),

            "src_ill": np.asarray(out["src_ill"], dtype=np.int64),
            "src_ont": np.asarray(out["src_ont"], dtype=np.int64),

            "match_cost": np.asarray(out["match_cost"], dtype=np.float32),
            "pair_id": np.asarray(out["pair_id"], dtype=np.int64),
            "row_weight": np.asarray(out["row_weight"], dtype=np.float32),
        }
        out_path = f"{args.out_prefix}.{args.mode}_{shard:04d}.npz"
        write_shard(out_path, payload)
        shard += 1
        for k in out:
            out[k].clear()

    def add_row(loc, mi, mo, i_idx, o_idx, cost, pid, weight):
        chrom, s, e = parse_locus(loc)
        out["locus"].append(loc)
        out["chrom"].append(chrom)
        out["locus_start"].append(s)
        out["locus_end"].append(e)

        out["mask_ill"].append(mi)
        out["mask_ont"].append(mo)

        if mi:
            out["emb_ill"].append(ill_emb[i_idx])
            out["logits_ill"].append(ill_lg[i_idx])
            out["probs_ill"].append(ill_pb[i_idx])
            out["src_ill"].append(i_idx)
        else:
            out["emb_ill"].append(np.zeros((D,), dtype=np.float32))
            out["logits_ill"].append(np.zeros((3,), dtype=np.float32))
            out["probs_ill"].append(np.zeros((3,), dtype=np.float32))
            out["src_ill"].append(-1)

        if mo:
            out["emb_ont"].append(ont_emb[o_idx])
            out["logits_ont"].append(ont_lg[o_idx])
            out["probs_ont"].append(ont_pb[o_idx])
            out["src_ont"].append(o_idx)
        else:
            out["emb_ont"].append(np.zeros((D,), dtype=np.float32))
            out["logits_ont"].append(np.zeros((3,), dtype=np.float32))
            out["probs_ont"].append(np.zeros((3,), dtype=np.float32))
            out["src_ont"].append(-1)

        out["match_cost"].append(cost)
        out["pair_id"].append(pid)
        out["row_weight"].append(weight)

        if len(out["locus"]) >= args.chunk_size:
            flush()

    # Build rows locus by locus
    for loc in loci_all:
        I = idx_ill.get(loc, [])
        O = idx_ont.get(loc, [])

        if not I and not O:
            continue

        if not I:
            # ONT-only
            for o in O:
                add_row(loc, 0, 1, -1, o, np.nan, -1, 1.0)
            continue

        if not O:
            # Illumina-only
            for i in I:
                add_row(loc, 1, 0, i, -1, np.nan, -1, 1.0)
            continue

        # both modalities present in this locus
        if args.no_match:
            # no matching: output all as unimodal but mark locus-level both via masks would require duplication
            # simplest: output unimodal rows only
            for i in I:
                add_row(loc, 1, 0, i, -1, np.nan, -1, 1.0)
            for o in O:
                add_row(loc, 0, 1, -1, o, np.nan, -1, 1.0)
            continue

        pairs, Iu, Ou = greedy_match_by_probs(I, O, ill_pb, ont_pb)

        if args.mode == "merged":
            # 1 fila por par + filas para no-matcheados
            for i, o, c in pairs:
                pid = pair_counter
                pair_counter += 1
                add_row(loc, 1, 1, i, o, c, pid, 1.0)
            for i in Iu:
                add_row(loc, 1, 0, i, -1, np.nan, -1, 1.0)
            for o in Ou:
                add_row(loc, 0, 1, -1, o, np.nan, -1, 1.0)

        else:
            # expanded: mantenemos todas las filas originales, y si hay match añadimos la otra modalidad
            # Creamos mapas i->(o,c,pid) y o->(i,c,pid)
            i2 = {}
            o2 = {}
            for i, o, c in pairs:
                pid = pair_counter
                pair_counter += 1
                i2[i] = (o, c, pid)
                o2[o] = (i, c, pid)

            for i in I:
                if i in i2:
                    o, c, pid = i2[i]
                    add_row(loc, 1, 1, i, o, c, pid, 0.5)
                else:
                    add_row(loc, 1, 0, i, -1, np.nan, -1, 1.0)

            for o in O:
                if o in o2:
                    i, c, pid = o2[o]
                    add_row(loc, 1, 1, i, o, c, pid, 0.5)
                else:
                    add_row(loc, 0, 1, -1, o, np.nan, -1, 1.0)

    flush()
    print("Done. mode=", args.mode, "shards=", shard)

if __name__ == "__main__":
    main()