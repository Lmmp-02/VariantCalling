#!/usr/bin/env python3
"""
Build INNER JOIN (1↔1 loci) and OUTER JOIN (singleton-only, excludes ambiguous loci) datasets
from sharded NPZ exports (Illumina + ONT).

Key fix vs previous version:
- NPZ access is grouped by shard pair (ill_npz, ont_npz) and arrays are loaded ONCE per shard.
  This avoids catastrophic repeated decompression of npz entries (label/embeddings/...).

INNER:
  - loci where ill_count==1 and ont_count==1
  - outputs rows with both modalities (mask_ill=1, mask_ont=1)

OUTER (singleton-only, excludes ambiguity):
  - loci where ill_count in {0,1} and ont_count in {0,1}
  - excludes any locus where ill_count>1 or ont_count>1
  - outputs rows with masks:
      (1,1) both present
      (1,0) only Illumina
      (0,1) only ONT
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


def iter_npz_paths(glob_pat: str) -> List[str]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"No NPZ matched: {glob_pat}")
    return paths


def count_loci(npz_paths: List[str]) -> Counter:
    c = Counter()
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        loci = d["locus"].astype(str)
        c.update(loci.tolist())
        d.close()
    return c


def get_dims(first_npz: str) -> Tuple[int, int, int]:
    d = np.load(first_npz, allow_pickle=True)
    emb_dim = int(d["embeddings"].shape[1])
    logits_dim = int(d["teacher_logits"].shape[1]) if "teacher_logits" in d.files else 0
    probs_dim = int(d["teacher_probs"].shape[1]) if "teacher_probs" in d.files else 0
    d.close()
    return emb_dim, logits_dim, probs_dim


def build_singleton_index(
    npz_paths: List[str],
    locus_counts: Counter,
    keep_loci: set,
) -> Dict[str, Tuple[str, int]]:
    """
    locus -> (npz_path, idx) for singleton loci (count==1) within keep_loci.
    """
    out: Dict[str, Tuple[str, int]] = {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        loci = d["locus"].astype(str)
        for i, L in enumerate(loci.tolist()):
            if L in keep_loci and locus_counts[L] == 1:
                out[L] = (p, i)
        d.close()
    return out


def new_buf(include_teacher: bool) -> dict:
    b = {
        "locus": [],
        "label": [],
        "mask_ill": [],
        "mask_ont": [],
        "group": [],
        "ill_embeddings": [],
        "ont_embeddings": [],
    }
    if include_teacher:
        b.update({
            "ill_logits": [],
            "ill_probs": [],
            "ont_logits": [],
            "ont_probs": [],
        })
    return b


def flush_npz(out_path: str, buf: dict, include_teacher: bool) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(buf["label"])

    payload = {
        "locus": np.asarray(buf["locus"], dtype=np.str_),
        "label": np.asarray(buf["label"], dtype=np.int64),
        "mask_ill": np.asarray(buf["mask_ill"], dtype=np.int64),
        "mask_ont": np.asarray(buf["mask_ont"], dtype=np.int64),
        "group": np.asarray(buf["group"], dtype=np.int64),
        "ill_embeddings": np.stack(buf["ill_embeddings"]).astype(np.float32),
        "ont_embeddings": np.stack(buf["ont_embeddings"]).astype(np.float32),
    }

    if include_teacher:
        payload.update({
            "ill_logits": np.stack(buf["ill_logits"]).astype(np.float32),
            "ill_probs": np.stack(buf["ill_probs"]).astype(np.float32),
            "ont_logits": np.stack(buf["ont_logits"]).astype(np.float32),
            "ont_probs": np.stack(buf["ont_probs"]).astype(np.float32),
        })

    np.savez_compressed(out_path, **payload)

    for k in buf:
        buf[k].clear()

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ill_glob", required=True)
    ap.add_argument("--ont_glob", required=True)

    ap.add_argument("--out_inner_dir", required=True)
    ap.add_argument("--out_outer_dir", required=True)
    ap.add_argument("--prefix", default="hg003_chr20")

    ap.add_argument("--chunk_size", type=int, default=20000)
    ap.add_argument("--drop_disagree", type=int, default=1,
                    help="INNER only: drop loci where labels disagree.")
    ap.add_argument("--include_teacher", type=int, default=1,
                    help="Include teacher logits/probs in outputs.")
    ap.add_argument("--write_meta_csv", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=2000,
                    help="Print progress every N rows kept.")
    ap.add_argument("--debug_shards", type=int, default=0,
                    help="If 1, print shard transitions (useful to see it's alive).")
    args = ap.parse_args()

    ill_paths = iter_npz_paths(args.ill_glob)
    ont_paths = iter_npz_paths(args.ont_glob)

    ill_emb_dim, ill_logits_dim, ill_probs_dim = get_dims(ill_paths[0])
    ont_emb_dim, ont_logits_dim, ont_probs_dim = get_dims(ont_paths[0])

    include_teacher = bool(args.include_teacher)
    if include_teacher:
        if ill_logits_dim == 0 or ont_logits_dim == 0:
            raise SystemExit("include_teacher=1 but teacher_logits missing in one modality NPZ.")
        if ill_probs_dim == 0 or ont_probs_dim == 0:
            raise SystemExit("include_teacher=1 but teacher_probs missing in one modality NPZ.")

    print("Counting loci (Illumina)...")
    ill_counts = count_loci(ill_paths)
    print("Counting loci (ONT)...")
    ont_counts = count_loci(ont_paths)

    ill_loci = set(ill_counts.keys())
    ont_loci = set(ont_counts.keys())

    inner_loci = {L for L in (ill_loci & ont_loci) if ill_counts[L] == 1 and ont_counts[L] == 1}

    outer_loci = set()
    for L in (ill_loci | ont_loci):
        ic = ill_counts.get(L, 0)
        oc = ont_counts.get(L, 0)
        if ic > 1 or oc > 1:
            continue
        if ic == 1 or oc == 1:
            outer_loci.add(L)

    print(f"INNER loci (1↔1): {len(inner_loci)}")
    print(f"OUTER loci (singleton-only, no ambiguous): {len(outer_loci)}")

    print("Building singleton locus index (Illumina)...")
    ill_single = build_singleton_index(ill_paths, ill_counts, outer_loci)
    print("Building singleton locus index (ONT)...")
    ont_single = build_singleton_index(ont_paths, ont_counts, outer_loci)

    # Zeros for missing modality (OUTER only)
    zero_ill_emb = np.zeros((ill_emb_dim,), dtype=np.float32)
    zero_ont_emb = np.zeros((ont_emb_dim,), dtype=np.float32)
    if include_teacher:
        zero_ill_logits = np.zeros((ill_logits_dim,), dtype=np.float32)
        zero_ill_probs = np.zeros((ill_probs_dim,), dtype=np.float32)
        zero_ont_logits = np.zeros((ont_logits_dim,), dtype=np.float32)
        zero_ont_probs = np.zeros((ont_probs_dim,), dtype=np.float32)

    inner_meta_rows = []
    outer_meta_rows = []

    # ---------------------------------------------------------
    # Build INNER (grouped by (ill_npz, ont_npz))
    # ---------------------------------------------------------
    print("\n==> Building INNER dataset (1↔1)...")

    # Build groups: (ipath, opath) -> list of (locus, iidx, oidx)
    inner_groups = defaultdict(list)
    missing_map = 0
    for L in inner_loci:
        if L not in ill_single or L not in ont_single:
            missing_map += 1
            continue
        ipath, iidx = ill_single[L]
        opath, oidx = ont_single[L]
        inner_groups[(ipath, opath)].append((L, iidx, oidx))

    group_keys = sorted(inner_groups.keys())
    if args.debug_shards:
        print(f"INNER shard-pairs: {len(group_keys)}")

    buf = new_buf(include_teacher)
    chunk = 0
    kept = 0
    dropped = 0

    for gi, (ipath, opath) in enumerate(group_keys, start=1):
        items = inner_groups[(ipath, opath)]

        if args.debug_shards and (gi == 1 or gi % 10 == 0):
            print(f"[INNER] loading shard-pair {gi}/{len(group_keys)}  ill={os.path.basename(ipath)}  ont={os.path.basename(opath)}  n_items={len(items)}")

        # Load BOTH shards and materialize arrays ONCE
        di = np.load(ipath, allow_pickle=True)
        do = np.load(opath, allow_pickle=True)

        ill_label = di["label"].astype(np.int64)
        ill_emb = di["embeddings"].astype(np.float32)
        if include_teacher:
            ill_logits = di["teacher_logits"].astype(np.float32)
            ill_probs = di["teacher_probs"].astype(np.float32)

        ont_label = do["label"].astype(np.int64)
        ont_emb = do["embeddings"].astype(np.float32)
        if include_teacher:
            ont_logits = do["teacher_logits"].astype(np.float32)
            ont_probs = do["teacher_probs"].astype(np.float32)

        for (L, iidx, oidx) in items:
            y_ill = int(ill_label[iidx])
            y_ont = int(ont_label[oidx])

            if args.drop_disagree and (y_ill != y_ont):
                dropped += 1
                continue

            y = y_ill

            buf["locus"].append(L)
            buf["label"].append(y)
            buf["mask_ill"].append(1)
            buf["mask_ont"].append(1)
            buf["group"].append(11)

            buf["ill_embeddings"].append(ill_emb[iidx])
            buf["ont_embeddings"].append(ont_emb[oidx])

            if include_teacher:
                buf["ill_logits"].append(ill_logits[iidx])
                buf["ill_probs"].append(ill_probs[iidx])
                buf["ont_logits"].append(ont_logits[oidx])
                buf["ont_probs"].append(ont_probs[oidx])

            if args.write_meta_csv:
                inner_meta_rows.append((L, y_ill, y_ont, ipath, iidx, opath, oidx))

            kept += 1
            if kept % args.chunk_size == 0:
                out = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_{chunk:04d}.npz")
                n = flush_npz(out, buf, include_teacher)
                print("Saved", out, "N=", n)
                chunk += 1

            if args.progress_every > 0 and kept % args.progress_every == 0:
                print(f"INNER progress: kept={kept} dropped={dropped} (current shard-pair {gi}/{len(group_keys)})")

        di.close()
        do.close()

    if buf["label"]:
        out = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_{chunk:04d}.npz")
        n = flush_npz(out, buf, include_teacher)
        print("Saved", out, "N=", n, "(last)")

    print(f"INNER done. kept={kept} dropped_disagree={dropped} missing_map={missing_map}")

    # ---------------------------------------------------------
    # Build OUTER (singleton-only) grouped by (ill_npz or '', ont_npz or '')
    # ---------------------------------------------------------
    print("\n==> Building OUTER dataset (singleton-only, no ambiguous)...")

    outer_groups = defaultdict(list)
    for L in outer_loci:
        has_ill = L in ill_single
        has_ont = L in ont_single
        ipath, iidx = ill_single[L] if has_ill else ("", -1)
        opath, oidx = ont_single[L] if has_ont else ("", -1)
        outer_groups[(ipath, opath)].append((L, iidx, oidx))

    group_keys = sorted(outer_groups.keys())
    if args.debug_shards:
        print(f"OUTER shard-pairs: {len(group_keys)}")

    buf = new_buf(include_teacher)
    chunk = 0
    kept = 0

    for gi, (ipath, opath) in enumerate(group_keys, start=1):
        items = outer_groups[(ipath, opath)]

        if args.debug_shards and (gi == 1 or gi % 10 == 0):
            print(f"[OUTER] loading shard-pair {gi}/{len(group_keys)}  ill={os.path.basename(ipath) if ipath else '-'}  ont={os.path.basename(opath) if opath else '-'}  n_items={len(items)}")

        # Load shards that exist, materialize arrays ONCE
        di = None
        do = None

        if ipath:
            di = np.load(ipath, allow_pickle=True)
            ill_label = di["label"].astype(np.int64)
            ill_emb = di["embeddings"].astype(np.float32)
            if include_teacher:
                ill_logits = di["teacher_logits"].astype(np.float32)
                ill_probs = di["teacher_probs"].astype(np.float32)

        if opath:
            do = np.load(opath, allow_pickle=True)
            ont_label = do["label"].astype(np.int64)
            ont_emb = do["embeddings"].astype(np.float32)
            if include_teacher:
                ont_logits = do["teacher_logits"].astype(np.float32)
                ont_probs = do["teacher_probs"].astype(np.float32)

        for (L, iidx, oidx) in items:
            mask_ill = 1 if ipath else 0
            mask_ont = 1 if opath else 0

            y_ill = int(ill_label[iidx]) if mask_ill else -1
            y_ont = int(ont_label[oidx]) if mask_ont else -1

            # label preference: Illumina if present else ONT
            y = y_ill if mask_ill else y_ont

            group = 11 if (mask_ill and mask_ont) else (10 if mask_ill else 1)  # 1 means 01

            buf["locus"].append(L)
            buf["label"].append(int(y))
            buf["mask_ill"].append(mask_ill)
            buf["mask_ont"].append(mask_ont)
            buf["group"].append(group)

            if mask_ill:
                buf["ill_embeddings"].append(ill_emb[iidx])
                if include_teacher:
                    buf["ill_logits"].append(ill_logits[iidx])
                    buf["ill_probs"].append(ill_probs[iidx])
            else:
                buf["ill_embeddings"].append(zero_ill_emb)
                if include_teacher:
                    buf["ill_logits"].append(zero_ill_logits)
                    buf["ill_probs"].append(zero_ill_probs)

            if mask_ont:
                buf["ont_embeddings"].append(ont_emb[oidx])
                if include_teacher:
                    buf["ont_logits"].append(ont_logits[oidx])
                    buf["ont_probs"].append(ont_probs[oidx])
            else:
                buf["ont_embeddings"].append(zero_ont_emb)
                if include_teacher:
                    buf["ont_logits"].append(zero_ont_logits)
                    buf["ont_probs"].append(zero_ont_probs)

            if args.write_meta_csv:
                outer_meta_rows.append((L, int(y), y_ill, y_ont, mask_ill, mask_ont, ipath, iidx, opath, oidx))

            kept += 1
            if kept % args.chunk_size == 0:
                out = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_{chunk:04d}.npz")
                n = flush_npz(out, buf, include_teacher)
                print("Saved", out, "N=", n)
                chunk += 1

            if args.progress_every > 0 and kept % args.progress_every == 0:
                print(f"OUTER progress: kept={kept} (current shard-pair {gi}/{len(group_keys)})")

        if di is not None:
            di.close()
        if do is not None:
            do.close()

    if buf["label"]:
        out = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_{chunk:04d}.npz")
        n = flush_npz(out, buf, include_teacher)
        print("Saved", out, "N=", n, "(last)")

    print(f"OUTER done. kept={kept}")

    # Meta CSVs
    if args.write_meta_csv:
        os.makedirs(args.out_inner_dir, exist_ok=True)
        os.makedirs(args.out_outer_dir, exist_ok=True)

        inner_csv = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_meta.csv")
        with open(inner_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["locus", "label_ill", "label_ont", "ill_npz", "ill_idx", "ont_npz", "ont_idx"])
            w.writerows(inner_meta_rows)
        print("Wrote", inner_csv)

        outer_csv = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_meta.csv")
        with open(outer_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["locus", "label", "label_ill", "label_ont", "mask_ill", "mask_ont",
                        "ill_npz", "ill_idx", "ont_npz", "ont_idx"])
            w.writerows(outer_meta_rows)
        print("Wrote", outer_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()