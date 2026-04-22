#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a single, reusable split_bins.json for OUTER (anti-leakage via genomic bins + buffer).

Key idea (fairness):
- The split is computed ONCE from the FULL OUTER set (all groups).
- Then ALL training scripts (ONT-only / Illumina-only / Hybrid) reuse the same bins.

Split method:
- bin_id = pos // bin_size
- choose contiguous TEST segment of bins
- choose contiguous VAL segment of bins, non-overlapping w.r.t TEST when expanded by buffer
- forbid bins in [lo-buffer, hi+buffer] around val/test segments (anti-leakage)
- train_bins = remaining bins
- dropped_bins = bins that fall inside forbidden range but are not in val_bins/test_bins

Outputs:
- split_bins.json (with bins lists + helpful metadata)
- optional printed summaries (counts by group/mask/labels for sanity)

Example:
python training/scripts/hybrid_dv/make_split_outer_bins.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --bin_size 1000000 --buffer_bins 1 --val_frac 0.1 --test_frac 0.1 \
  --seed 0 \
  --save_dir training/out/splits/outer_chr20_v1
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Locus -> pos parsing
# -----------------------------
_POS_RE = re.compile(r"(?:chr)?([0-9XYM]+)[\:\-_](\d+)", re.IGNORECASE)

def parse_locus_pos(locus: str) -> Optional[int]:
    m = _POS_RE.search(locus)
    if m:
        return int(m.group(2))
    # fallback: grab first integer chunk
    toks = re.split(r"[^0-9]+", locus)
    toks = [t for t in toks if t]
    if toks:
        return int(toks[0])
    return None


def list_shards(outer_dir: Path) -> List[Path]:
    shards = sorted(outer_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {outer_dir}")
    return shards


# -----------------------------
# Split: contiguous segments + buffer
# -----------------------------
def split_bins_contiguous(
    all_bins: List[int],
    seed: int,
    val_frac: float,
    test_frac: float,
    buffer_bins: int,
) -> Tuple[set, set, set, set]:
    """
    Returns (train_bins, val_bins, test_bins, dropped_bins)
    using contiguous bin segments + buffer exclusion to prevent leakage.

    IMPORTANT: all_bins must be sorted unique list of bin_ids present in data.
    """
    rng = random.Random(seed)
    n = len(all_bins)
    if n == 0:
        raise ValueError("No bins to split.")

    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))

    if n_test + n_val >= n:
        raise RuntimeError("val_frac + test_frac too large for number of bins.")

    def pick_segment(length: int, forbidden_range: Optional[Tuple[int, int]]):
        max_start = n - length
        starts = list(range(0, max_start + 1))
        rng.shuffle(starts)
        for s in starts:
            seg = all_bins[s:s + length]
            lo, hi = seg[0], seg[-1]
            exp_lo, exp_hi = lo - buffer_bins, hi + buffer_bins
            if forbidden_range is None:
                return set(seg), (exp_lo, exp_hi)

            f_lo, f_hi = forbidden_range
            # non-overlap between expanded ranges
            if exp_hi < f_lo or exp_lo > f_hi:
                return set(seg), (exp_lo, exp_hi)

        raise RuntimeError(
            "Cannot find non-overlapping val/test segments. "
            "Reduce val/test fracs or buffer_bins."
        )

    # Pick TEST first, then VAL avoiding TEST expanded range
    test_bins, test_exp = pick_segment(n_test, None)
    val_bins,  val_exp  = pick_segment(n_val, test_exp)

    # Build forbidden bins (expanded ranges)
    forbidden = set()
    for lo, hi in [test_exp, val_exp]:
        forbidden.update(range(lo, hi + 1))

    train_bins = set([b for b in all_bins if b not in forbidden])
    dropped_bins = set([b for b in all_bins if (b in forbidden and b not in (val_bins | test_bins))])

    return train_bins, val_bins, test_bins, dropped_bins


# -----------------------------
# Summaries (optional sanity)
# -----------------------------
def summarize_subset(
    name: str,
    bin_ids: np.ndarray,
    y3: np.ndarray,
    group: np.ndarray,
    mask_ill: np.ndarray,
    mask_ont: np.ndarray,
    bins_set: set,
) -> Dict:
    sel = np.isin(bin_ids, np.fromiter(bins_set, dtype=np.int64))
    n = int(sel.sum())
    if n == 0:
        return {"name": name, "n": 0}

    y3s = y3[sel]
    ybin = (y3s != 0).astype(np.int64)
    gs = group[sel]
    mi = mask_ill[sel]
    mo = mask_ont[sel]

    def _count_vals(arr, keys):
        out = {}
        for k in keys:
            out[int(k)] = int((arr == k).sum())
        return out

    by_group = _count_vals(gs, [1, 10, 11])
    by_y3 = _count_vals(y3s, [0, 1, 2])
    by_ybin = _count_vals(ybin, [0, 1])

    # mask combos as strings "01"/"10"/"11"
    mkey = np.char.add(mi.astype(str), mo.astype(str))
    by_mask = {
        "01": int((mkey == "01").sum()),
        "10": int((mkey == "10").sum()),
        "11": int((mkey == "11").sum()),
    }

    return {
        "name": name,
        "n": n,
        "by_group": by_group,
        "by_mask": by_mask,
        "by_ybin": by_ybin,
        "by_y3": by_y3,
    }


def print_summary(s: Dict) -> None:
    n = max(int(s.get("n", 0)), 1)
    def pct(x): return 100.0 * float(x) / float(n)
    print(f"\n[SplitSummary] {s.get('name','?')} n={int(s.get('n',0)):,}")
    if "by_group" in s:
        print("  groups:", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in s["by_group"].items()})
    if "by_mask" in s:
        print("  masks: ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in s["by_mask"].items()})
    if "by_ybin" in s:
        print("  y_bin:", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in s["by_ybin"].items()})
    if "by_y3" in s:
        print("  y_3:  ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in s["by_y3"].items()})


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer_dir", type=str, required=True,
                    help="Directory with OUTER shards (*.npz).")
    ap.add_argument("--save_dir", type=str, required=True,
                    help="Where to write split_bins.json (and optional stats).")

    ap.add_argument("--bin_size", type=int, default=1_000_000)
    ap.add_argument("--buffer_bins", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no_print", action="store_true",
                    help="Disable printing split summaries.")
    args = ap.parse_args()

    outer_dir = Path(args.outer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = list_shards(outer_dir)
    print(f"[Index] Scanning {len(shards)} shards in {outer_dir} ...")

    # Collect minimal arrays
    bin_ids: List[int] = []
    y3s: List[int] = []
    groups: List[int] = []
    mi_list: List[int] = []
    mo_list: List[int] = []
    n_skipped = 0

    for si, sp in enumerate(shards, 1):
        npz = np.load(sp, allow_pickle=True)

        loci = npz["locus"]
        if loci.dtype.kind in ("S", "O"):
            loci_list = [
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in loci.tolist()
            ]
        else:
            loci_list = [str(x) for x in loci.tolist()]

        y = npz["label"].astype(np.int64).reshape(-1)
        grp = npz["group"].astype(np.int64).reshape(-1)
        mi = npz["mask_ill"].astype(np.int64).reshape(-1)
        mo = npz["mask_ont"].astype(np.int64).reshape(-1)

        for i, L in enumerate(loci_list):
            pos = parse_locus_pos(L)
            if pos is None:
                n_skipped += 1
                continue
            b = pos // int(args.bin_size)
            bin_ids.append(int(b))
            y3s.append(int(y[i]))
            groups.append(int(grp[i]))
            mi_list.append(int(mi[i]))
            mo_list.append(int(mo[i]))

        npz.close()

        if si % 10 == 0 or si == len(shards):
            print(f"  - {si:>4}/{len(shards)} shards indexed")

    if not bin_ids:
        raise SystemExit("No examples indexed (locus parsing failed?).")

    bin_ids_arr = np.asarray(bin_ids, dtype=np.int64)
    y3_arr = np.asarray(y3s, dtype=np.int64)
    group_arr = np.asarray(groups, dtype=np.int64)
    mi_arr = np.asarray(mi_list, dtype=np.int64)
    mo_arr = np.asarray(mo_list, dtype=np.int64)

    all_bins = sorted(set(bin_ids_arr.tolist()))
    train_bins, val_bins, test_bins, dropped_bins = split_bins_contiguous(
        all_bins=all_bins,
        seed=int(args.seed),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        buffer_bins=int(args.buffer_bins),
    )

    # extra diagnostics: missing bins between min/max
    minb, maxb = int(min(all_bins)), int(max(all_bins))
    present = set(all_bins)
    missing_bins = [b for b in range(minb, maxb + 1) if b not in present]

    payload = {
        "outer_dir": str(outer_dir),
        "n_shards": int(len(shards)),
        "n_examples_indexed": int(len(bin_ids_arr)),
        "n_skipped_locus_parse": int(n_skipped),

        "bin_size": int(args.bin_size),
        "buffer_bins": int(args.buffer_bins),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),

        "train_bins": sorted(list(train_bins)),
        "val_bins": sorted(list(val_bins)),
        "test_bins": sorted(list(test_bins)),
        "dropped_bins": sorted(list(dropped_bins)),

        "all_bins_present": all_bins,
        "missing_bins_between_min_max": missing_bins,
    }

    out_json = save_dir / "split_bins.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[Split] bins: train={len(train_bins)} val={len(val_bins)} test={len(test_bins)} dropped_for_buffer={len(dropped_bins)}")
    print(f"[Saved] {out_json}")

    if not args.no_print:
        tr = summarize_subset("train", bin_ids_arr, y3_arr, group_arr, mi_arr, mo_arr, train_bins)
        va = summarize_subset("val",   bin_ids_arr, y3_arr, group_arr, mi_arr, mo_arr, val_bins)
        te = summarize_subset("test",  bin_ids_arr, y3_arr, group_arr, mi_arr, mo_arr, test_bins)
        print_summary(tr)
        print_summary(va)
        print_summary(te)


if __name__ == "__main__":
    main()