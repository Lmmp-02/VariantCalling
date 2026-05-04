#!/usr/bin/env python3
"""
Build INNER JOIN (1↔1 loci) and/or OUTER JOIN (singleton-only, excludes ambiguous loci)
datasets from sharded NPZ exports (Illumina + ONT).

Key properties:
- NPZ access is grouped by shard pair (ill_npz, ont_npz) and arrays are loaded ONCE per shard.
- Can build INNER, OUTER, or both.
- OUTER keeps singleton loci only and excludes ambiguous loci when requested.
- Writes chrom/locus_start/locus_end for downstream chrom-based splits.
- Optionally propagates VCF-oriented metadata for downstream calling/VCF generation.
- Can drop ambiguous bimodal loci consistently in both INNER and OUTER:
  cases where Illumina and ONT disagree in label and/or variant_type.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def iter_npz_paths(glob_pat: str) -> List[str]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"No NPZ matched: {glob_pat}")
    return paths


def parse_locus(locus: str) -> Tuple[str, int, int]:
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", locus)
    if not m:
        return ("", -1, -1)
    return (m.group(1), int(m.group(2)), int(m.group(3)))


VCF_STR_FIELDS = [
    "vcf_chrom",
    "vcf_ref",
    "vcf_alt",
    "vcf_alt_full",
    "variant_key",
    "variant_key_full",
]

VCF_INT_FIELDS = [
    "vcf_pos",
    "vcf_start0",
    "vcf_end0",
    "variant_encoded_len",
]

VCF_FIELDS = VCF_STR_FIELDS + VCF_INT_FIELDS


def empty_vcf_meta_from_locus(locus: str) -> dict:
    chrom, s, e = parse_locus(locus)
    return {
        "vcf_chrom": chrom,
        "vcf_pos": int(s) if s >= 0 else -1,
        "vcf_start0": int(s - 1) if s >= 1 else -1,
        "vcf_end0": int(e) if e >= 0 else -1,
        "vcf_ref": "",
        "vcf_alt": "",
        "vcf_alt_full": "",
        "variant_key": "",
        "variant_key_full": "",
        "variant_encoded_len": 0,
    }


def load_vcf_arrays(d: np.lib.npyio.NpzFile) -> dict:
    """
    Load VCF-oriented arrays from a unimodal NPZ shard.

    Missing fields are represented as None so the join remains backwards-compatible
    with legacy NPZs that do not yet contain VCF metadata.
    """
    out = {}
    files = set(d.files)

    for k in VCF_STR_FIELDS:
        out[k] = d[k].astype(str) if k in files else None

    for k in VCF_INT_FIELDS:
        out[k] = d[k].astype(np.int64) if k in files else None

    return out


def row_vcf_meta(vcf_arrays: dict, idx: int, locus: str) -> dict:
    """
    Extract one row of VCF metadata from preloaded arrays.
    Falls back to empty metadata when fields are missing.
    """
    meta = empty_vcf_meta_from_locus(locus)

    if vcf_arrays is None:
        return meta

    for k in VCF_STR_FIELDS:
        arr = vcf_arrays.get(k)
        if arr is not None:
            meta[k] = str(arr[idx])

    for k in VCF_INT_FIELDS:
        arr = vcf_arrays.get(k)
        if arr is not None:
            meta[k] = int(arr[idx])

    return meta


def vcf_meta_has_key(m: dict) -> bool:
    return bool(str(m.get("variant_key", "")).strip())


def vcf_key_mismatch(ill_meta: dict, ont_meta: dict) -> int:
    ki = str(ill_meta.get("variant_key", "")).strip()
    ko = str(ont_meta.get("variant_key", "")).strip()
    if not ki or not ko:
        return 0
    return int(ki != ko)


def select_merged_vcf_meta(
    *,
    mask_ill: int,
    mask_ont: int,
    ill_meta: dict,
    ont_meta: dict,
    source_policy: str,
) -> Tuple[dict, str, int]:
    """
    Select canonical merged VCF metadata for the joined row.

    The join key remains `locus`; this only decides which VCF fields are exposed
    as the merged/canonical fields for downstream calling.

    source_policy:
      - prefer_ill: if Illumina VCF key exists, use Illumina, else ONT
      - prefer_ont: if ONT VCF key exists, use ONT, else Illumina
      - ill: always use Illumina when present
      - ont: always use ONT when present
    """
    mismatch = 0

    if mask_ill and mask_ont:
        mismatch = vcf_key_mismatch(ill_meta, ont_meta)

        if source_policy == "prefer_ont":
            if vcf_meta_has_key(ont_meta):
                return ont_meta, "both_prefer_ont", mismatch
            return ill_meta, "both_fallback_ill", mismatch

        if source_policy == "ont":
            return ont_meta, "both_ont", mismatch

        if source_policy == "ill":
            return ill_meta, "both_ill", mismatch

        # default: prefer_ill
        if vcf_meta_has_key(ill_meta):
            return ill_meta, "both_prefer_ill", mismatch
        return ont_meta, "both_fallback_ont", mismatch

    if mask_ill:
        return ill_meta, "ill", 0

    if mask_ont:
        return ont_meta, "ont", 0

    return empty_vcf_meta_from_locus(""), "none", 0


def append_vcf_meta(
    buf: dict,
    *,
    merged: dict,
    ill_meta: dict,
    ont_meta: dict,
    source: str,
    mismatch: int,
) -> None:
    """
    Append merged VCF metadata plus modality-specific VCF metadata.
    """
    for k in VCF_STR_FIELDS:
        buf[k].append(str(merged.get(k, "")))
        buf[f"ill_{k}"].append(str(ill_meta.get(k, "")))
        buf[f"ont_{k}"].append(str(ont_meta.get(k, "")))

    for k in VCF_INT_FIELDS:
        buf[k].append(int(merged.get(k, -1 if k != "variant_encoded_len" else 0)))
        buf[f"ill_{k}"].append(int(ill_meta.get(k, -1 if k != "variant_encoded_len" else 0)))
        buf[f"ont_{k}"].append(int(ont_meta.get(k, -1 if k != "variant_encoded_len" else 0)))

    buf["vcf_source"].append(str(source))
    buf["vcf_key_mismatch"].append(int(mismatch))


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
    """locus -> (npz_path, idx) for singleton loci (count==1) within keep_loci."""
    out: Dict[str, Tuple[str, int]] = {}
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        loci = d["locus"].astype(str)
        for i, L in enumerate(loci.tolist()):
            if L in keep_loci and locus_counts[L] == 1:
                out[L] = (p, i)
        d.close()
    return out


def new_buf(include_teacher: bool, include_vcf_meta: bool) -> dict:
    b = {
        "locus": [],
        "chrom": [],
        "locus_start": [],
        "locus_end": [],
        "label": [],
        "variant_type": [],
        "variant_type_mismatch": [],
        "ill_variant_type": [],
        "ont_variant_type": [],
        "mask_ill": [],
        "mask_ont": [],
        "group": [],
        "ill_embeddings": [],
        "ont_embeddings": [],
    }

    if include_vcf_meta:
        for k in VCF_FIELDS:
            b[k] = []
            b[f"ill_{k}"] = []
            b[f"ont_{k}"] = []

        b["vcf_source"] = []
        b["vcf_key_mismatch"] = []

    if include_teacher:
        b.update({
            "ill_logits": [],
            "ill_probs": [],
            "ont_logits": [],
            "ont_probs": [],
        })

    return b


def flush_npz(out_path: str, buf: dict, include_teacher: bool, include_vcf_meta: bool) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(buf["label"])

    payload = {
        "locus": np.asarray(buf["locus"], dtype=np.str_),
        "chrom": np.asarray(buf["chrom"], dtype=np.str_),
        "locus_start": np.asarray(buf["locus_start"], dtype=np.int64),
        "locus_end": np.asarray(buf["locus_end"], dtype=np.int64),
        "label": np.asarray(buf["label"], dtype=np.int64),
        "variant_type": np.asarray(buf["variant_type"], dtype=np.int64),
        "variant_type_mismatch": np.asarray(buf["variant_type_mismatch"], dtype=np.int64),
        "ill_variant_type": np.asarray(buf["ill_variant_type"], dtype=np.int64),
        "ont_variant_type": np.asarray(buf["ont_variant_type"], dtype=np.int64),
        "mask_ill": np.asarray(buf["mask_ill"], dtype=np.int64),
        "mask_ont": np.asarray(buf["mask_ont"], dtype=np.int64),
        "group": np.asarray(buf["group"], dtype=np.int64),
        "ill_embeddings": np.stack(buf["ill_embeddings"]).astype(np.float32),
        "ont_embeddings": np.stack(buf["ont_embeddings"]).astype(np.float32),
    }

    if include_vcf_meta:
        for k in VCF_STR_FIELDS:
            payload[k] = np.asarray(buf[k], dtype=np.str_)
            payload[f"ill_{k}"] = np.asarray(buf[f"ill_{k}"], dtype=np.str_)
            payload[f"ont_{k}"] = np.asarray(buf[f"ont_{k}"], dtype=np.str_)

        for k in VCF_INT_FIELDS:
            payload[k] = np.asarray(buf[k], dtype=np.int64)
            payload[f"ill_{k}"] = np.asarray(buf[f"ill_{k}"], dtype=np.int64)
            payload[f"ont_{k}"] = np.asarray(buf[f"ont_{k}"], dtype=np.int64)

        payload["vcf_source"] = np.asarray(buf["vcf_source"], dtype=np.str_)
        payload["vcf_key_mismatch"] = np.asarray(buf["vcf_key_mismatch"], dtype=np.int64)

    if include_teacher:
        payload.update({
            "ill_logits": np.stack(buf["ill_logits"]).astype(np.float32),
            "ill_probs": np.stack(buf["ill_probs"]).astype(np.float32),
            "ont_logits": np.stack(buf["ont_logits"]).astype(np.float32),
            "ont_probs": np.stack(buf["ont_probs"]).astype(np.float32),
        })

    # Defensive consistency check.
    bad = {}
    for k, v in payload.items():
        if hasattr(v, "shape") and len(v.shape) >= 1 and int(v.shape[0]) != n:
            bad[k] = int(v.shape[0])

    if bad:
        raise RuntimeError(f"Payload length mismatch before saving {out_path}: expected={n}, bad={bad}")

    np.savez_compressed(out_path, **payload)

    for k in buf:
        buf[k].clear()

    return n


def append_common_meta(buf: dict, locus: str):
    chrom, s, e = parse_locus(locus)
    buf["locus"].append(locus)
    buf["chrom"].append(chrom)
    buf["locus_start"].append(s)
    buf["locus_end"].append(e)


def resolve_variant_type(mask_ill: int, mask_ont: int, ill_vt: int, ont_vt: int) -> Tuple[int, int]:
    """
    Return:
      merged_variant_type, mismatch_flag

    Rules:
    - only Illumina present -> ill_variant_type
    - only ONT present -> ont_variant_type
    - both present and equal -> that value
    - both present and different -> -1 and mismatch=1
    """
    if mask_ill and mask_ont:
        if ill_vt == ont_vt:
            return int(ill_vt), 0
        return -1, 1
    if mask_ill:
        return int(ill_vt), 0
    if mask_ont:
        return int(ont_vt), 0
    return -1, 0


def make_meta_csv_row(
    *,
    locus: str,
    y: int,
    y_ill: int,
    y_ont: int,
    vt: int,
    vt_ill: int,
    vt_ont: int,
    vt_mismatch: int,
    mask_ill: int,
    mask_ont: int,
    ipath: str,
    iidx: int,
    opath: str,
    oidx: int,
    include_vcf_meta: bool,
    merged_vcf_meta: dict | None = None,
    ill_vcf_meta: dict | None = None,
    ont_vcf_meta: dict | None = None,
    vcf_source: str = "",
    vcf_mismatch: int = 0,
) -> list:
    row = [
        locus,
        int(y),
        int(y_ill),
        int(y_ont),
        int(vt),
        int(vt_ill),
        int(vt_ont),
        int(vt_mismatch),
        int(mask_ill),
        int(mask_ont),
        ipath,
        int(iidx),
        opath,
        int(oidx),
    ]

    if include_vcf_meta:
        merged_vcf_meta = merged_vcf_meta or {}
        ill_vcf_meta = ill_vcf_meta or {}
        ont_vcf_meta = ont_vcf_meta or {}

        row.extend([
            merged_vcf_meta.get("vcf_chrom", ""),
            int(merged_vcf_meta.get("vcf_pos", -1)),
            merged_vcf_meta.get("vcf_ref", ""),
            merged_vcf_meta.get("vcf_alt", ""),
            merged_vcf_meta.get("vcf_alt_full", ""),
            merged_vcf_meta.get("variant_key", ""),
            merged_vcf_meta.get("variant_key_full", ""),
            vcf_source,
            int(vcf_mismatch),
            ill_vcf_meta.get("variant_key", ""),
            ont_vcf_meta.get("variant_key", ""),
        ])

    return row


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ill_glob", required=True)
    ap.add_argument("--ont_glob", required=True)

    ap.add_argument("--out_inner_dir", required=True)
    ap.add_argument("--out_outer_dir", required=True)
    ap.add_argument("--prefix", default="hg003_chr20")

    ap.add_argument("--chunk_size", type=int, default=20000)
    ap.add_argument("--drop_ambiguous_bimodal", type=int, default=1,
                    help="If 1, drop bimodal loci where Illumina and ONT disagree in label and/or variant_type.")
    ap.add_argument("--include_teacher", type=int, default=1,
                    help="Include teacher logits/probs in outputs.")
    ap.add_argument("--write_meta_csv", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=2000,
                    help="Print progress every N rows kept.")
    ap.add_argument("--debug_shards", type=int, default=0,
                    help="If 1, print shard transitions.")
    ap.add_argument("--build_inner", type=int, default=1)
    ap.add_argument("--build_outer", type=int, default=1)
    ap.add_argument("--include_vcf_meta", type=int, default=1,
                    help="If 1, propagate VCF metadata fields when available in unimodal NPZs.")
    ap.add_argument("--vcf_meta_source", type=str, default="prefer_ill",
                    choices=["prefer_ill", "prefer_ont", "ill", "ont"],
                    help="Canonical merged VCF metadata source for bimodal rows.")
    ap.add_argument("--drop_vcf_key_mismatch", type=int, default=0,
                    help="If 1, drop bimodal rows where Illumina and ONT variant_key differ.")
    args = ap.parse_args()

    build_inner = bool(args.build_inner)
    build_outer = bool(args.build_outer)
    include_teacher = bool(args.include_teacher)
    drop_ambiguous_bimodal = bool(args.drop_ambiguous_bimodal)
    include_vcf_meta = bool(args.include_vcf_meta)
    drop_vcf_key_mismatch = bool(args.drop_vcf_key_mismatch)

    if not build_inner and not build_outer:
        raise SystemExit("Nothing to build: set --build_inner 1 and/or --build_outer 1.")

    ill_paths = iter_npz_paths(args.ill_glob)
    ont_paths = iter_npz_paths(args.ont_glob)

    ill_emb_dim, ill_logits_dim, ill_probs_dim = get_dims(ill_paths[0])
    ont_emb_dim, ont_logits_dim, ont_probs_dim = get_dims(ont_paths[0])

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
    print(f"OUTER loci (singleton-only before ambiguity filter): {len(outer_loci)}")

    keep_loci = outer_loci if build_outer else inner_loci
    print("Building singleton locus index (Illumina)...")
    ill_single = build_singleton_index(ill_paths, ill_counts, keep_loci)
    print("Building singleton locus index (ONT)...")
    ont_single = build_singleton_index(ont_paths, ont_counts, keep_loci)

    zero_ill_emb = np.zeros((ill_emb_dim,), dtype=np.float32)
    zero_ont_emb = np.zeros((ont_emb_dim,), dtype=np.float32)
    if include_teacher:
        zero_ill_logits = np.zeros((ill_logits_dim,), dtype=np.float32)
        zero_ill_probs = np.zeros((ill_probs_dim,), dtype=np.float32)
        zero_ont_logits = np.zeros((ont_logits_dim,), dtype=np.float32)
        zero_ont_probs = np.zeros((ont_probs_dim,), dtype=np.float32)

    inner_meta_rows = []
    outer_meta_rows = []

    if build_inner:
        print("\n==> Building INNER dataset (1↔1)...")
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

        buf = new_buf(include_teacher, include_vcf_meta)
        chunk = 0
        kept = 0
        candidates = 0
        dropped_ambiguous = 0
        dropped_label_mismatch = 0
        dropped_variant_type_mismatch = 0
        observed_label_mismatch = 0
        observed_variant_type_mismatch = 0
        observed_vcf_key_mismatch = 0
        dropped_vcf_key_mismatch = 0        


        for gi, (ipath, opath) in enumerate(group_keys, start=1):
            items = inner_groups[(ipath, opath)]
            if args.debug_shards and (gi == 1 or gi % 10 == 0):
                print(f"[INNER] loading shard-pair {gi}/{len(group_keys)} ill={os.path.basename(ipath)} ont={os.path.basename(opath)} n_items={len(items)}")

            di = np.load(ipath, allow_pickle=True)
            do = np.load(opath, allow_pickle=True)

            if include_vcf_meta:
                ill_vcf_arrays = load_vcf_arrays(di)
                ont_vcf_arrays = load_vcf_arrays(do)            

            ill_label = di["label"].astype(np.int64)
            ill_emb = di["embeddings"].astype(np.float32)
            ill_variant_type = di["variant_type"].astype(np.int64) if "variant_type" in di.files else np.full_like(ill_label, -1)
            if include_teacher:
                ill_logits = di["teacher_logits"].astype(np.float32)
                ill_probs = di["teacher_probs"].astype(np.float32)

            ont_label = do["label"].astype(np.int64)
            ont_emb = do["embeddings"].astype(np.float32)
            ont_variant_type = do["variant_type"].astype(np.int64) if "variant_type" in do.files else np.full_like(ont_label, -1)
            if include_teacher:
                ont_logits = do["teacher_logits"].astype(np.float32)
                ont_probs = do["teacher_probs"].astype(np.float32)

            for (L, iidx, oidx) in items:
                candidates += 1

                y_ill = int(ill_label[iidx])
                y_ont = int(ont_label[oidx])
                vt_ill = int(ill_variant_type[iidx])
                vt_ont = int(ont_variant_type[oidx])

                label_mismatch = y_ill != y_ont
                variant_type_mismatch = vt_ill != vt_ont

                if label_mismatch:
                    observed_label_mismatch += 1
                if variant_type_mismatch:
                    observed_variant_type_mismatch += 1

                if include_vcf_meta:
                    ill_vcf_meta = row_vcf_meta(ill_vcf_arrays, iidx, L)
                    ont_vcf_meta = row_vcf_meta(ont_vcf_arrays, oidx, L)

                    merged_vcf_meta, vcf_source, vcf_mismatch = select_merged_vcf_meta(
                        mask_ill=1,
                        mask_ont=1,
                        ill_meta=ill_vcf_meta,
                        ont_meta=ont_vcf_meta,
                        source_policy=args.vcf_meta_source,
                    )
                else:
                    vcf_mismatch = 0

                if include_vcf_meta and vcf_mismatch:
                    observed_vcf_key_mismatch += 1

                if drop_ambiguous_bimodal and (label_mismatch or variant_type_mismatch):
                    dropped_ambiguous += 1
                    if label_mismatch:
                        dropped_label_mismatch += 1
                    if variant_type_mismatch:
                        dropped_variant_type_mismatch += 1
                    continue

                if include_vcf_meta and drop_vcf_key_mismatch and vcf_mismatch:
                    dropped_vcf_key_mismatch += 1
                    continue

                y = y_ill
                vt, vt_mismatch = resolve_variant_type(1, 1, vt_ill, vt_ont)

                append_common_meta(buf, L)
                buf["label"].append(y)
                buf["variant_type"].append(vt)
                buf["variant_type_mismatch"].append(vt_mismatch)
                buf["ill_variant_type"].append(vt_ill)
                buf["ont_variant_type"].append(vt_ont)
                buf["mask_ill"].append(1)
                buf["mask_ont"].append(1)
                buf["group"].append(11)
                buf["ill_embeddings"].append(ill_emb[iidx])
                buf["ont_embeddings"].append(ont_emb[oidx])

                if include_vcf_meta:
                    append_vcf_meta(
                        buf,
                        merged=merged_vcf_meta,
                        ill_meta=ill_vcf_meta,
                        ont_meta=ont_vcf_meta,
                        source=vcf_source,
                        mismatch=vcf_mismatch,
                    )

                if include_teacher:
                    buf["ill_logits"].append(ill_logits[iidx])
                    buf["ill_probs"].append(ill_probs[iidx])
                    buf["ont_logits"].append(ont_logits[oidx])
                    buf["ont_probs"].append(ont_probs[oidx])

                if args.write_meta_csv:
                    inner_meta_rows.append(
                        make_meta_csv_row(
                            locus=L,
                            y=int(y),
                            y_ill=y_ill,
                            y_ont=y_ont,
                            vt=vt,
                            vt_ill=vt_ill,
                            vt_ont=vt_ont,
                            vt_mismatch=vt_mismatch,
                            mask_ill=1,
                            mask_ont=1,
                            ipath=ipath,
                            iidx=iidx,
                            opath=opath,
                            oidx=oidx,
                            include_vcf_meta=include_vcf_meta,
                            merged_vcf_meta=merged_vcf_meta if include_vcf_meta else None,
                            ill_vcf_meta=ill_vcf_meta if include_vcf_meta else None,
                            ont_vcf_meta=ont_vcf_meta if include_vcf_meta else None,
                            vcf_source=vcf_source if include_vcf_meta else "",
                            vcf_mismatch=vcf_mismatch if include_vcf_meta else 0,
                        )
                    )

                kept += 1
                if kept % args.chunk_size == 0:
                    out = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_{chunk:04d}.npz")
                    n = flush_npz(out, buf, include_teacher, include_vcf_meta)
                    print("Saved", out, "N=", n)
                    chunk += 1

                if args.progress_every > 0 and kept % args.progress_every == 0:
                    print(
                        f"INNER progress: kept={kept} "
                        f"dropped_ambiguous={dropped_ambiguous} "
                        f"(current shard-pair {gi}/{len(group_keys)})"
                    )

            di.close()
            do.close()

        if buf["label"]:
            out = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_{chunk:04d}.npz")
            n = flush_npz(out, buf, include_teacher, include_vcf_meta)
            print("Saved", out, "N=", n, "(last)")

        inner_report = {
            "build_inner": True,
            "drop_ambiguous_bimodal": drop_ambiguous_bimodal,
            "inner_loci_1to1": len(inner_loci),
            "missing_map": missing_map,
            "candidate_rows": candidates,
            "kept_rows": kept,
            "dropped_ambiguous_bimodal": dropped_ambiguous,
            "dropped_label_mismatch": dropped_label_mismatch,
            "dropped_variant_type_mismatch": dropped_variant_type_mismatch,
            "observed_label_mismatch": observed_label_mismatch,
            "observed_variant_type_mismatch": observed_variant_type_mismatch,
            "include_vcf_meta": include_vcf_meta,
            "vcf_meta_source": args.vcf_meta_source,
            "drop_vcf_key_mismatch": drop_vcf_key_mismatch,
            "observed_vcf_key_mismatch": observed_vcf_key_mismatch,
            "dropped_vcf_key_mismatch": dropped_vcf_key_mismatch,
        }
        write_json(os.path.join(args.out_inner_dir, f"{args.prefix}_inner_report.json"), inner_report)

        print(
            f"INNER done. kept={kept} "
            f"dropped_ambiguous={dropped_ambiguous} "
            f"missing_map={missing_map}"
        )

    if build_outer:
        print("\n==> Building OUTER dataset (singleton-only)...")
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

        buf = new_buf(include_teacher, include_vcf_meta)
        chunk = 0
        kept = 0
        candidates = 0
        bimodal_candidates = 0
        ill_only_candidates = 0
        ont_only_candidates = 0
        dropped_ambiguous = 0
        dropped_label_mismatch = 0
        dropped_variant_type_mismatch = 0
        observed_label_mismatch = 0
        observed_variant_type_mismatch = 0
        observed_vcf_key_mismatch = 0
        dropped_vcf_key_mismatch = 0        

        for gi, (ipath, opath) in enumerate(group_keys, start=1):
            items = outer_groups[(ipath, opath)]
            if args.debug_shards and (gi == 1 or gi % 10 == 0):
                print(f"[OUTER] loading shard-pair {gi}/{len(group_keys)} ill={os.path.basename(ipath) if ipath else '-'} ont={os.path.basename(opath) if opath else '-'} n_items={len(items)}")

            di = None
            do = None
            if ipath:
                di = np.load(ipath, allow_pickle=True)
                if include_vcf_meta:
                    ill_vcf_arrays = load_vcf_arrays(di)
                ill_label = di["label"].astype(np.int64)
                ill_emb = di["embeddings"].astype(np.float32)
                ill_variant_type = di["variant_type"].astype(np.int64) if "variant_type" in di.files else np.full_like(ill_label, -1)
                if include_teacher:
                    ill_logits = di["teacher_logits"].astype(np.float32)
                    ill_probs = di["teacher_probs"].astype(np.float32)
            if opath:
                do = np.load(opath, allow_pickle=True)
                if include_vcf_meta:
                    ont_vcf_arrays = load_vcf_arrays(do)
                ont_label = do["label"].astype(np.int64)
                ont_emb = do["embeddings"].astype(np.float32)
                ont_variant_type = do["variant_type"].astype(np.int64) if "variant_type" in do.files else np.full_like(ont_label, -1)
                if include_teacher:
                    ont_logits = do["teacher_logits"].astype(np.float32)
                    ont_probs = do["teacher_probs"].astype(np.float32)

            if include_vcf_meta and not ipath:
                ill_vcf_arrays = None
            if include_vcf_meta and not opath:
                ont_vcf_arrays = None

            for (L, iidx, oidx) in items:
                candidates += 1

                mask_ill = 1 if ipath else 0
                mask_ont = 1 if opath else 0

                if mask_ill and mask_ont:
                    bimodal_candidates += 1
                elif mask_ill:
                    ill_only_candidates += 1
                elif mask_ont:
                    ont_only_candidates += 1

                y_ill = int(ill_label[iidx]) if mask_ill else -1
                y_ont = int(ont_label[oidx]) if mask_ont else -1
                vt_ill = int(ill_variant_type[iidx]) if mask_ill else -1
                vt_ont = int(ont_variant_type[oidx]) if mask_ont else -1

                if include_vcf_meta:
                    ill_vcf_meta = (
                        row_vcf_meta(ill_vcf_arrays, iidx, L)
                        if mask_ill else empty_vcf_meta_from_locus(L)
                    )
                    ont_vcf_meta = (
                        row_vcf_meta(ont_vcf_arrays, oidx, L)
                        if mask_ont else empty_vcf_meta_from_locus(L)
                    )

                    merged_vcf_meta, vcf_source, vcf_mismatch = select_merged_vcf_meta(
                        mask_ill=mask_ill,
                        mask_ont=mask_ont,
                        ill_meta=ill_vcf_meta,
                        ont_meta=ont_vcf_meta,
                        source_policy=args.vcf_meta_source,
                    )
                else:
                    vcf_mismatch = 0

                label_mismatch = (mask_ill and mask_ont and y_ill != y_ont)
                variant_type_mismatch = (mask_ill and mask_ont and vt_ill != vt_ont)

                if label_mismatch:
                    observed_label_mismatch += 1
                if variant_type_mismatch:
                    observed_variant_type_mismatch += 1

                if include_vcf_meta and vcf_mismatch:
                    observed_vcf_key_mismatch += 1

                if drop_ambiguous_bimodal and (label_mismatch or variant_type_mismatch):
                    dropped_ambiguous += 1
                    if label_mismatch:
                        dropped_label_mismatch += 1
                    if variant_type_mismatch:
                        dropped_variant_type_mismatch += 1
                    continue

                if include_vcf_meta and drop_vcf_key_mismatch and vcf_mismatch:
                    dropped_vcf_key_mismatch += 1
                    continue

                y = y_ill if mask_ill else y_ont
                group = 11 if (mask_ill and mask_ont) else (10 if mask_ill else 1)
                vt, vt_mismatch = resolve_variant_type(mask_ill, mask_ont, vt_ill, vt_ont)

                append_common_meta(buf, L)
                buf["label"].append(int(y))
                buf["variant_type"].append(vt)
                buf["variant_type_mismatch"].append(vt_mismatch)
                buf["ill_variant_type"].append(vt_ill)
                buf["ont_variant_type"].append(vt_ont)
                buf["mask_ill"].append(mask_ill)
                buf["mask_ont"].append(mask_ont)
                buf["group"].append(group)

                if include_vcf_meta:
                    append_vcf_meta(
                        buf,
                        merged=merged_vcf_meta,
                        ill_meta=ill_vcf_meta,
                        ont_meta=ont_vcf_meta,
                        source=vcf_source,
                        mismatch=vcf_mismatch,
                    )

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
                    outer_meta_rows.append(
                        make_meta_csv_row(
                            locus=L,
                            y=int(y),
                            y_ill=y_ill,
                            y_ont=y_ont,
                            vt=vt,
                            vt_ill=vt_ill,
                            vt_ont=vt_ont,
                            vt_mismatch=vt_mismatch,
                            mask_ill=mask_ill,
                            mask_ont=mask_ont,
                            ipath=ipath,
                            iidx=iidx,
                            opath=opath,
                            oidx=oidx,
                            include_vcf_meta=include_vcf_meta,
                            merged_vcf_meta=merged_vcf_meta if include_vcf_meta else None,
                            ill_vcf_meta=ill_vcf_meta if include_vcf_meta else None,
                            ont_vcf_meta=ont_vcf_meta if include_vcf_meta else None,
                            vcf_source=vcf_source if include_vcf_meta else "",
                            vcf_mismatch=vcf_mismatch if include_vcf_meta else 0,
                        )
                    )

                kept += 1
                if kept % args.chunk_size == 0:
                    out = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_{chunk:04d}.npz")
                    n = flush_npz(out, buf, include_teacher, include_vcf_meta)
                    print("Saved", out, "N=", n)
                    chunk += 1

                if args.progress_every > 0 and kept % args.progress_every == 0:
                    print(
                        f"OUTER progress: kept={kept} "
                        f"dropped_ambiguous={dropped_ambiguous} "
                        f"(current shard-pair {gi}/{len(group_keys)})"
                    )

            if di is not None:
                di.close()
            if do is not None:
                do.close()

        if buf["label"]:
            out = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_{chunk:04d}.npz")
            n = flush_npz(out, buf, include_teacher, include_vcf_meta)
            print("Saved", out, "N=", n, "(last)")

        outer_report = {
            "build_outer": True,
            "drop_ambiguous_bimodal": drop_ambiguous_bimodal,
            "outer_loci_singleton_pre_filter": len(outer_loci),
            "candidate_rows": candidates,
            "bimodal_candidates": bimodal_candidates,
            "ill_only_candidates": ill_only_candidates,
            "ont_only_candidates": ont_only_candidates,
            "kept_rows": kept,
            "dropped_ambiguous_bimodal": dropped_ambiguous,
            "dropped_label_mismatch": dropped_label_mismatch,
            "dropped_variant_type_mismatch": dropped_variant_type_mismatch,
            "observed_label_mismatch": observed_label_mismatch,
            "observed_variant_type_mismatch": observed_variant_type_mismatch,
            "include_vcf_meta": include_vcf_meta,
            "vcf_meta_source": args.vcf_meta_source,
            "drop_vcf_key_mismatch": drop_vcf_key_mismatch,
            "observed_vcf_key_mismatch": observed_vcf_key_mismatch,
            "dropped_vcf_key_mismatch": dropped_vcf_key_mismatch,
        }
        write_json(os.path.join(args.out_outer_dir, f"{args.prefix}_outer_report.json"), outer_report)

        print(
            f"OUTER done. kept={kept} "
            f"dropped_ambiguous={dropped_ambiguous}"
        )

    if args.write_meta_csv:
        header = [
            "locus",
            "label",
            "label_ill",
            "label_ont",
            "variant_type",
            "ill_variant_type",
            "ont_variant_type",
            "variant_type_mismatch",
            "mask_ill",
            "mask_ont",
            "ill_npz",
            "ill_idx",
            "ont_npz",
            "ont_idx",
        ]

        if include_vcf_meta:
            header.extend([
                "vcf_chrom",
                "vcf_pos",
                "vcf_ref",
                "vcf_alt",
                "vcf_alt_full",
                "variant_key",
                "variant_key_full",
                "vcf_source",
                "vcf_key_mismatch",
                "ill_variant_key",
                "ont_variant_key",
            ])

        if build_inner:
            os.makedirs(args.out_inner_dir, exist_ok=True)
            inner_csv = os.path.join(args.out_inner_dir, f"{args.prefix}_inner_meta.csv")
            with open(inner_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(inner_meta_rows)
            print("Wrote", inner_csv)

        if build_outer:
            os.makedirs(args.out_outer_dir, exist_ok=True)
            outer_csv = os.path.join(args.out_outer_dir, f"{args.prefix}_outer_meta.csv")
            with open(outer_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(outer_meta_rows)
            print("Wrote", outer_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()