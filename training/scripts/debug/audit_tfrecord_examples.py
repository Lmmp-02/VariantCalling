#!/usr/bin/env python3
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

import tensorflow as tf


def parse_one(rec_bytes):
    ex = tf.train.Example.FromString(rec_bytes)
    f = ex.features.feature

    label = int(f["label"].int64_list.value[0]) if "label" in f else -1
    variant_type = int(f["variant_type"].int64_list.value[0]) if "variant_type" in f else -1
    sequencing_type = int(f["sequencing_type"].int64_list.value[0]) if "sequencing_type" in f else -1

    has_variant_encoded = int("variant/encoded" in f)
    variant_encoded_len = (
        len(f["variant/encoded"].bytes_list.value[0])
        if "variant/encoded" in f and f["variant/encoded"].bytes_list.value
        else 0
    )

    has_alt_indices = int("alt_allele_indices/encoded" in f)

    return {
        "label": label,
        "truth_variant_binary": int(label > 0),
        "variant_type_code": variant_type,
        "sequencing_type": sequencing_type,
        "has_variant_encoded": has_variant_encoded,
        "variant_encoded_len": variant_encoded_len,
        "has_alt_indices": has_alt_indices,
    }


def audit(name, tfrecord_glob):
    paths = sorted(glob.glob(tfrecord_glob))
    if not paths:
        raise SystemExit("No TFRecords matched for {}: {}".format(name, tfrecord_glob))

    counts = Counter()
    total = 0

    ds = tf.data.TFRecordDataset(paths, compression_type="GZIP")

    for rec in ds:
        d = parse_one(rec.numpy())
        total += 1

        counts[("label", d["label"])] += 1
        counts[("variant_type_code", d["variant_type_code"])] += 1
        counts[("variant_type_code__label", d["variant_type_code"], d["label"], d["truth_variant_binary"])] += 1
        counts[("has_variant_encoded", d["has_variant_encoded"])] += 1
        counts[("has_alt_indices", d["has_alt_indices"])] += 1

        if total % 100000 == 0:
            print("{}: processed {}".format(name, total), flush=True)

    summary = {
        "dataset": name,
        "tfrecord_glob": tfrecord_glob,
        "tfrecord_files": len(paths),
        "rows": total,
    }

    rows = []
    for key, count in sorted(counts.items(), key=lambda x: str(x[0])):
        if key[0] == "label":
            rows.append({
                "dataset": name,
                "kind": "label",
                "a": key[1],
                "b": "",
                "c": "",
                "count": count,
            })
        elif key[0] == "variant_type_code":
            rows.append({
                "dataset": name,
                "kind": "variant_type_code",
                "a": key[1],
                "b": "",
                "c": "",
                "count": count,
            })
        elif key[0] == "variant_type_code__label":
            rows.append({
                "dataset": name,
                "kind": "variant_type_code__label",
                "a": key[1],
                "b": key[2],
                "c": key[3],
                "count": count,
            })
        elif key[0] == "has_variant_encoded":
            rows.append({
                "dataset": name,
                "kind": "has_variant_encoded",
                "a": key[1],
                "b": "",
                "c": "",
                "count": count,
            })
        elif key[0] == "has_alt_indices":
            rows.append({
                "dataset": name,
                "kind": "has_alt_indices",
                "a": key[1],
                "b": "",
                "c": "",
                "count": count,
            })

    return summary, rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True, help="NAME=TFRECORD_GLOB")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    count_rows = []

    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit("Bad --dataset spec: {}. Expected NAME=TFRECORD_GLOB".format(spec))

        name, glob_s = spec.split("=", 1)

        print("\n=== Auditing raw TFRecords: {} ===".format(name), flush=True)
        s, rows = audit(name, glob_s)
        summaries.append(s)
        count_rows.extend(rows)

    write_csv(
        out_dir / "tfrecord_audit_summary.csv",
        summaries,
        ["dataset", "tfrecord_glob", "tfrecord_files", "rows"],
    )

    write_csv(
        out_dir / "tfrecord_audit_counts.csv",
        count_rows,
        ["dataset", "kind", "a", "b", "c", "count"],
    )

    print("\n=== Summary ===")
    for s in summaries:
        print(s)

    print("\nWrote:")
    print(out_dir / "tfrecord_audit_summary.csv")
    print(out_dir / "tfrecord_audit_counts.csv")


if __name__ == "__main__":
    main()
