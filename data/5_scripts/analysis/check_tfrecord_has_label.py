#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from typing import Any

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect gzipped DeepVariant TFRecords and perform structural sanity checks."
    )
    p.add_argument("--glob", required=True, help="Glob pattern for TFRecord shards (*.gz).")
    p.add_argument(
        "--max_records",
        type=int,
        default=3,
        help="Maximum number of records to inspect in total.",
    )
    p.add_argument(
        "--require_label",
        type=int,
        choices=[0, 1],
        default=1,
        help="Require that the inspected records contain the 'label' feature.",
    )
    p.add_argument(
        "--expected_width",
        type=int,
        default=0,
        help="Expected pileup image width. 0 disables this check.",
    )
    p.add_argument(
        "--expected_channels",
        type=int,
        default=0,
        help="Expected number of channels. 0 disables this check.",
    )
    return p.parse_args()


def _get_int64_list(feature: Any) -> list[int]:
    if feature is None:
        return []
    return list(feature.int64_list.value)


def _get_bytes(feature: Any) -> bytes | None:
    if feature is None or not feature.bytes_list.value:
        return None
    return feature.bytes_list.value[0]


def _decode_bytes_maybe(b: bytes | None) -> str | None:
    if b is None:
        return None
    try:
        return b.decode("utf-8")
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    pattern = args.glob
    max_records = max(1, int(args.max_records))

    files = sorted(glob.glob(pattern))
    print(f"glob={pattern}")
    print(f"files_found={len(files)}")

    if not files:
        print("files_inspected=0")
        print("records_checked=0")
        print("validation_status=error")
        print("reason=no_files_matched")
        return 2

    records_checked = 0
    files_inspected = 0
    label_found = False
    first_record_feature_keys: list[str] | None = None
    unique_shapes: set[tuple[int, ...]] = set()
    loci_seen: list[str] = []
    widths_seen: set[int] = set()
    channels_seen: set[int] = set()
    errors: list[str] = []

    for fpath in files:
        files_inspected += 1
        try:
            ds = tf.data.TFRecordDataset([fpath], compression_type="GZIP")
        except Exception as exc:
            errors.append(f"error_opening_file={fpath}: {exc}")
            continue

        for raw in ds:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            feats = ex.features.feature
            feature_keys = sorted(feats.keys())

            if first_record_feature_keys is None:
                first_record_feature_keys = feature_keys
                print("first_record_feature_keys=" + ",".join(feature_keys))

            # image shape
            shape_vals = _get_int64_list(feats.get("image/shape"))
            if shape_vals:
                shape_t = tuple(shape_vals)
                unique_shapes.add(shape_t)
                if len(shape_vals) >= 2:
                    widths_seen.add(shape_vals[1])
                if len(shape_vals) >= 3:
                    channels_seen.add(shape_vals[2])

            # locus
            locus = _decode_bytes_maybe(_get_bytes(feats.get("locus")))
            if locus is not None and len(loci_seen) < 5:
                loci_seen.append(locus)

            # label
            if "label" in feats:
                label_found = True

            records_checked += 1
            if records_checked >= max_records:
                break

        if records_checked >= max_records:
            break

    print(f"files_inspected={files_inspected}")
    print(f"records_checked={records_checked}")
    print(f"label_found={label_found}")

    if first_record_feature_keys is None:
        print("validation_status=error")
        print("reason=no_records_read")
        for e in errors:
            print(e)
        return 3

    if unique_shapes:
        print(
            "unique_image_shapes="
            + ";".join("x".join(map(str, s)) for s in sorted(unique_shapes))
        )

    if loci_seen:
        print("example_loci=" + ",".join(loci_seen))

    # checks
    validation_ok = True

    if args.require_label == 1 and not label_found:
        validation_ok = False
        print("check_label=FAIL")
    else:
        print("check_label=OK")

    if args.expected_width > 0:
        if widths_seen == {args.expected_width}:
            print(f"check_width=OK ({args.expected_width})")
        else:
            validation_ok = False
            print(f"check_width=FAIL expected={args.expected_width} seen={sorted(widths_seen)}")

    if args.expected_channels > 0:
        if channels_seen == {args.expected_channels}:
            print(f"check_channels=OK ({args.expected_channels})")
        else:
            validation_ok = False
            print(
                f"check_channels=FAIL expected={args.expected_channels} seen={sorted(channels_seen)}"
            )

    if errors:
        for e in errors:
            print(e)

    if validation_ok:
        print("validation_status=ok")
        return 0
    else:
        print("validation_status=warning")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())