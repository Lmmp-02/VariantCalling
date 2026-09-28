#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a meta CSV for OUTER join that includes variant_type (SNP/INDEL) per example.

Input (existing):
  data/4_out/datasets/HG003/join_outer_singletons/hg003_chr20_outer.meta.csv
    columns typically:
      locus,label,label_ill,label_ont,mask_ill,mask_ont,ill_npz,ill_idx,ont_npz,ont_idx

Output (new CSV):
  adds:
    - group (1/10/11 derived from masks)
    - bin_id (pos//bin_size)
    - split (train/val/test/dropped/unknown) if --split_bins_json provided
    - variant_type_ill, variant_type_ont, variant_type (final), vt_name
    - vt_mismatch (1 if both present and disagree)

We try to read variant_type from the referenced unimodal npz:
  keys searched: variant_type, vartype, varianttype, type
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

_POS_RE = re.compile(r"(?:chr)?([0-9XYM]+)[\:\-_](\d+)", re.IGNORECASE)

def parse_locus_pos(locus: str) -> Optional[int]:
    m = _POS_RE.search(locus)
    if m:
        return int(m.group(2))
    toks = re.split(r"[^0-9]+", locus)
    toks = [t for t in toks if t]
    if toks:
        return int(toks[0])
    return None

def vt_name(vt: int) -> str:
    if vt == 1:
        return "SNP"
    if vt == 2:
        return "INDEL"
    return "UNKNOWN"

def infer_group(mask_ill: int, mask_ont: int) -> int:
    if mask_ill == 0 and mask_ont == 1:
        return 1
    if mask_ill == 1 and mask_ont == 0:
        return 10
    if mask_ill == 1 and mask_ont == 1:
        return 11
    # unexpected
    return -1

def find_vt_key(npz_obj) -> Optional[str]:
    for k in npz_obj.files:
        kl = k.lower()
        if kl in ("variant_type", "vartype", "varianttype", "type"):
            return k
    return None

class VtReader:
    """
    Caches variant_type arrays per npz path to avoid repeated disk IO.
    """
    def __init__(self):
        self.cache: Dict[str, Tuple[Optional[np.ndarray], Optional[str]]] = {}

    def get_vt(self, npz_path: str, idx: int) -> int:
        if idx is None or idx < 0:
            return 0
        if not npz_path:
            return 0

        p = Path(npz_path)
        if not p.exists():
            return 0

        key = str(p)
        if key not in self.cache:
            npz = np.load(key, allow_pickle=True, mmap_mode="r")
            k = find_vt_key(npz)
            arr = None
            if k is not None:
                try:
                    arr = npz[k]
                except Exception:
                    arr = None
            # keep only the array reference (mmap) + key; close npz wrapper
            npz.close()
            self.cache[key] = (arr, k)

        arr, _k = self.cache[key]
        if arr is None:
            return 0
        if idx >= len(arr):
            return 0
        v = arr[idx]
        if isinstance(v, np.generic):
            v = v.item()
        try:
            v = int(v)
        except Exception:
            return 0
        return v if v in (1, 2) else 0

def load_split_bins(split_bins_json: Path) -> Optional[Dict[str, Any]]:
    if split_bins_json is None:
        return None
    if not split_bins_json.exists():
        return None
    return json.loads(split_bins_json.read_text(encoding="utf-8"))

def split_of_bin(bin_id: int, split_payload: Dict[str, Any]) -> str:
    if split_payload is None:
        return "unknown"
    if bin_id in set(split_payload.get("train_bins", [])):
        return "train"
    if bin_id in set(split_payload.get("val_bins", [])):
        return "val"
    if bin_id in set(split_payload.get("test_bins", [])):
        return "test"
    if bin_id in set(split_payload.get("dropped_bins", [])):
        return "dropped"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True,
                    help="Path to hg003_chr20_outer.meta.csv")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV with variant_type columns")
    ap.add_argument("--split_bins_json", type=str, default=None,
                    help="Optional split_bins.json to add split=train/val/test")
    ap.add_argument("--bin_size", type=int, default=1_000_000)

    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_bins(Path(args.split_bins_json)) if args.split_bins_json else None
    vt_reader = VtReader()

    # counters for sanity
    cnt_total = 0
    cnt_vt = {0: 0, 1: 0, 2: 0}
    cnt_by_split = {}
    cnt_by_group = {}
    cnt_mismatch = 0

    with open(in_csv, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames_in = reader.fieldnames or []
        required = {"locus", "label", "mask_ill", "mask_ont", "ill_npz", "ill_idx", "ont_npz", "ont_idx"}
        missing = [c for c in required if c not in set(fieldnames_in)]
        if missing:
            raise SystemExit(f"[ERROR] Missing required columns in input CSV: {missing}")

        extra_cols = [
            "group", "bin_id", "split",
            "variant_type_ill", "variant_type_ont", "variant_type", "vt_name", "vt_mismatch"
        ]
        fieldnames_out = fieldnames_in + [c for c in extra_cols if c not in set(fieldnames_in)]

        with open(out_csv, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
            writer.writeheader()

            for row in reader:
                locus = row["locus"]
                pos = parse_locus_pos(locus)
                if pos is None:
                    # skip rows we cannot bin
                    continue
                bin_id = pos // int(args.bin_size)

                mask_ill = int(row["mask_ill"])
                mask_ont = int(row["mask_ont"])
                group = infer_group(mask_ill, mask_ont)

                # read vt from referenced unimodal npzs
                ill_npz = (row.get("ill_npz") or "").strip()
                ont_npz = (row.get("ont_npz") or "").strip()
                ill_idx = int(row.get("ill_idx") or -1)
                ont_idx = int(row.get("ont_idx") or -1)

                vt_ill = vt_reader.get_vt(ill_npz, ill_idx) if ill_npz and ill_idx >= 0 else 0
                vt_ont = vt_reader.get_vt(ont_npz, ont_idx) if ont_npz and ont_idx >= 0 else 0

                mismatch = 0
                if vt_ill in (1, 2) and vt_ont in (1, 2) and vt_ill != vt_ont:
                    mismatch = 1

                # choose final vt (prefer agreement; otherwise prefer the modality that exists for that group)
                if vt_ill in (1, 2) and vt_ont in (1, 2):
                    vt_final = vt_ill if vt_ill == vt_ont else (vt_ill if group in (10, 11) else vt_ont)
                elif vt_ill in (1, 2):
                    vt_final = vt_ill
                elif vt_ont in (1, 2):
                    vt_final = vt_ont
                else:
                    vt_final = 0

                sp = split_of_bin(bin_id, split_payload) if split_payload else "unknown"

                # write
                row["group"] = str(group)
                row["bin_id"] = str(bin_id)
                row["split"] = sp
                row["variant_type_ill"] = str(vt_ill)
                row["variant_type_ont"] = str(vt_ont)
                row["variant_type"] = str(vt_final)
                row["vt_name"] = vt_name(vt_final)
                row["vt_mismatch"] = str(mismatch)

                writer.writerow(row)

                # stats
                cnt_total += 1
                cnt_vt[vt_final] = cnt_vt.get(vt_final, 0) + 1
                cnt_by_split[sp] = cnt_by_split.get(sp, 0) + 1
                cnt_by_group[group] = cnt_by_group.get(group, 0) + 1
                cnt_mismatch += mismatch

    print(f"[Done] wrote: {out_csv}")
    print(f"[Stats] n={cnt_total:,}")
    print(f"[Stats] by_group: { {k: cnt_by_group[k] for k in sorted(cnt_by_group)} }")
    print(f"[Stats] by_split: { {k: cnt_by_split[k] for k in sorted(cnt_by_split)} }")
    print(f"[Stats] variant_type: {{0/UNK:{cnt_vt.get(0,0):,}, 1/SNP:{cnt_vt.get(1,0):,}, 2/INDEL:{cnt_vt.get(2,0):,}}}")
    print(f"[Stats] vt_mismatch (ill vs ont): {cnt_mismatch:,}")

if __name__ == "__main__":
    main()