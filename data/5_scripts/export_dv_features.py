#!/usr/bin/env python3
"""export_dv_features.py

Export DeepVariant model features (embeddings/logits/probs) from make_examples TFRecords.

Designed for DeepVariant docker images (e.g. google/deepvariant:1.9.0) and their
SavedModel bundles under /opt/models/*.

This version avoids decoding variant protobufs (no variants_pb2, no tf.parse_tensor).

Default outputs per .npz shard:
  - embeddings:      (N, D) float32 (typically D=2048)
  - teacher_logits:  (N, 3) float32
  - teacher_probs:   (N, 3) float32
  - label:           (N,) int64 (=-1 if not present in TFRecords)
  - locus:           (N,) str   (e.g. chr20:88865-88867)

  - alt_idx_list:    (N,) str   e.g. "0", "1", "0,1" (decoded from alt_allele_indices/encoded)
  - alt_idx0:        (N,) int64 first alt index (for convenience)

  - chrom:           (N,) str   from locus
  - locus_start:     (N,) int64
  - locus_end:       (N,) int64
  - variant_type:    (N,) int64 (from TFExample if present else -1)
  - sequencing_type: (N,) int64 (from TFExample if present else -1)

Optional identity metadata (disabled by default, enable with --emit_identity_meta 1):
  - variant_hash:    (N,) str   sha1(variant/encoded) hex
  - example_id:      (N,) str   f"{variant_hash}:{alt_idx_list}"
  - variant_len:     (N,) int64 len(variant/encoded)

Usage (inside DeepVariant docker):
  python3 -u /data/scripts/export_dv_features.py \
    --model_dir /opt/models/wgs \
    --tfrecord_glob "data/.../make_examples.tfrecord-*-of-00014.gz" \
    --out_prefix "data/.../hg003_chr20_illumina"

If tensor names differ across models, pass --emb_tensor/--logits_tensor.
Otherwise the script tries to auto-detect suitable tensors.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# Known-good defaults for many DeepVariant SavedModels (InceptionV3-based).
DEFAULT_EMB_NAME = "StatefulPartitionedCall/inceptionv3/dropout/Identity:0"
DEFAULT_LOGITS_NAME = "StatefulPartitionedCall/inceptionv3/classification/BiasAdd:0"


def _parse_locus(locus: str) -> Tuple[str, int, int]:
    """Parse locus like 'chr20:88865-88867' -> (chrom, start, end)."""
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", locus)
    if not m:
        return ("", -1, -1)
    return (m.group(1), int(m.group(2)), int(m.group(3)))


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    """Decode protobuf varint from buf starting at i -> (value, next_i)."""
    shift = 0
    x = 0
    while True:
        if i >= len(buf):
            raise ValueError("Truncated varint")
        b = buf[i]
        i += 1
        x |= (b & 0x7F) << shift
        if not (b & 0x80):
            return x, i
        shift += 7
        if shift > 70:
            raise ValueError("Varint too long")


def decode_alt_indices_encoded(ab: bytes) -> List[int]:
    """Decode DeepVariant 'alt_allele_indices/encoded' which is protobuf packed varints.

    Example blobs seen:
      0a0100      -> [0]
      0a0101      -> [1]
      0a020001    -> [0, 1]
    """
    if not ab:
        return [0]

    # Expect field 1 (0x0A) wire type 2 (len-delimited)
    if ab[0] != 0x0A:
        # Fallback: interpret first byte as value
        return [int(ab[0])]

    try:
        n, j = _read_varint(ab, 1)
        payload = ab[j : j + n]
        vals: List[int] = []
        i = 0
        while i < len(payload):
            v, i = _read_varint(payload, i)
            vals.append(int(v))
        return vals if vals else [0]
    except Exception:
        return [0]


def _sha1_hex(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _empty_vcf_meta(chrom_l: str, s_l: int, e_l: int) -> dict:
    return {
        "vcf_chrom": chrom_l,
        "vcf_pos": int(s_l + 1) if s_l >= 0 else -1,
        "vcf_start0": int(s_l),
        "vcf_end0": int(e_l),
        "vcf_ref": "",
        "vcf_alt": "",
        "vcf_alt_full": "",
        "variant_key": "",
        "variant_key_full": "",
        "variant_encoded_len": 0,
    }


def _make_variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    if not chrom or pos < 1 or not ref or not alt:
        return ""
    return f"{chrom}:{pos}:{ref}>{alt}"


def _read_len_delimited(buf: bytes, i: int) -> Tuple[bytes, int]:
    n, i = _read_varint(buf, i)
    j = i + n
    if j > len(buf):
        raise ValueError("Truncated length-delimited protobuf field")
    return buf[i:j], j


def _skip_proto_field(buf: bytes, i: int, wire_type: int) -> int:
    """
    Skip an unknown protobuf field.

    Wire types:
      0 = varint
      1 = 64-bit
      2 = length-delimited
      5 = 32-bit
    """
    if wire_type == 0:
        _, i = _read_varint(buf, i)
        return i

    if wire_type == 1:
        j = i + 8
        if j > len(buf):
            raise ValueError("Truncated 64-bit protobuf field")
        return j

    if wire_type == 2:
        _, j = _read_len_delimited(buf, i)
        return j

    if wire_type == 5:
        j = i + 4
        if j > len(buf):
            raise ValueError("Truncated 32-bit protobuf field")
        return j

    raise ValueError(f"Unsupported protobuf wire_type={wire_type}")


def decode_nucleus_variant_minimal(variant_bytes: bytes) -> dict:
    """
    Minimal manual decoder for Nucleus Variant protobuf.

    Only decodes the fields needed for VCF reconstruction.

    Nucleus Variant field numbers:
      reference_name    = 14
      start             = 16
      end               = 13
      reference_bases   = 6
      alternate_bases   = 7

    This intentionally avoids third_party.nucleus.protos.variants_pb2.
    """
    out = {
        "reference_name": "",
        "start": -1,
        "end": -1,
        "reference_bases": "",
        "alternate_bases": [],
    }

    i = 0
    n = len(variant_bytes)

    while i < n:
        key, i = _read_varint(variant_bytes, i)
        field_num = key >> 3
        wire_type = key & 0x07

        # string reference_name = 14;
        if field_num == 14 and wire_type == 2:
            b, i = _read_len_delimited(variant_bytes, i)
            out["reference_name"] = b.decode("utf-8", errors="replace")

        # int64 start = 16;
        elif field_num == 16 and wire_type == 0:
            v, i = _read_varint(variant_bytes, i)
            out["start"] = int(v)

        # int64 end = 13;
        elif field_num == 13 and wire_type == 0:
            v, i = _read_varint(variant_bytes, i)
            out["end"] = int(v)

        # string reference_bases = 6;
        elif field_num == 6 and wire_type == 2:
            b, i = _read_len_delimited(variant_bytes, i)
            out["reference_bases"] = b.decode("utf-8", errors="replace")

        # repeated string alternate_bases = 7;
        elif field_num == 7 and wire_type == 2:
            b, i = _read_len_delimited(variant_bytes, i)
            out["alternate_bases"].append(b.decode("utf-8", errors="replace"))

        else:
            i = _skip_proto_field(variant_bytes, i, wire_type)

    return out


def decode_variant_vcf_meta(
    variant_bytes: bytes,
    alt_indices: List[int],
    chrom_l: str,
    s_l: int,
    e_l: int,
    strict: bool = True,
) -> dict:
    """
    Decode DeepVariant variant/encoded into VCF-oriented metadata.

    This version does NOT depend on Nucleus / variants_pb2.

    VCF POS is 1-based.
    DeepVariant/Nucleus Variant.start is 0-based.
    """
    if not variant_bytes:
        if strict:
            raise RuntimeError("Missing variant/encoded while --emit_vcf_meta=1.")
        return _empty_vcf_meta(chrom_l, s_l, e_l)

    try:
        v = decode_nucleus_variant_minimal(variant_bytes)

        chrom = str(v["reference_name"])
        start0 = int(v["start"])
        end0 = int(v["end"])
        pos = start0 + 1 if start0 >= 0 else -1
        ref = str(v["reference_bases"])
        all_alts = [str(a) for a in v["alternate_bases"]]

        if not chrom or pos < 1 or end0 < 0 or not ref or not all_alts:
            if strict:
                raise RuntimeError(
                    "Decoded variant is missing required VCF fields: "
                    f"chrom={chrom!r}, start0={start0}, end0={end0}, "
                    f"ref={ref!r}, alternate_bases={all_alts!r}"
                )
            return _empty_vcf_meta(chrom_l, s_l, e_l)

        selected_alts: List[str] = []
        for idx in alt_indices:
            if 0 <= int(idx) < len(all_alts):
                selected_alts.append(all_alts[int(idx)])

        if not selected_alts:
            if strict:
                raise RuntimeError(
                    f"Could not map alt_indices={alt_indices} to alternate_bases={all_alts}"
                )
            selected_alts = all_alts

        alt = ",".join(selected_alts)
        alt_full = ",".join(all_alts)

        return {
            "vcf_chrom": chrom,
            "vcf_pos": pos,
            "vcf_start0": start0,
            "vcf_end0": end0,
            "vcf_ref": ref,
            "vcf_alt": alt,
            "vcf_alt_full": alt_full,
            "variant_key": _make_variant_key(chrom, pos, ref, alt),
            "variant_key_full": _make_variant_key(chrom, pos, ref, alt_full),
            "variant_encoded_len": len(variant_bytes),
        }

    except Exception:
        if strict:
            raise
        return _empty_vcf_meta(chrom_l, s_l, e_l)


def parse_example(
    rec_bytes: bytes,
    emit_identity_meta: bool = False,
    emit_vcf_meta: bool = True,
    strict_vcf_meta: bool = True,
):
    """Parse a DeepVariant TF.train.Example -> image (float32), label (int), locus(str), meta (dict)."""
    ex = tf.train.Example.FromString(rec_bytes)
    f = ex.features.feature

    # Image
    img_bytes = f["image/encoded"].bytes_list.value[0]
    shape = list(f["image/shape"].int64_list.value)  # e.g. [100,221,7]
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape).astype(np.float32)
    # DeepVariant typical normalization: uint8 -> float in [-1, 1)
    img = img / 128.0 - 1.0

    # Label is present in training-mode TFRecords; absent in calling-mode.
    y = int(f["label"].int64_list.value[0]) if "label" in f else -1
    locus = f["locus"].bytes_list.value[0].decode("utf-8") if "locus" in f else ""

    chrom_l, s_l, e_l = _parse_locus(locus)

    # alt indices bytes (packed varints)
    ab = f["alt_allele_indices/encoded"].bytes_list.value[0] if "alt_allele_indices/encoded" in f else b""
    alt_list = decode_alt_indices_encoded(ab)
    alt_list_str = ",".join(str(x) for x in alt_list)
    alt0 = int(alt_list[0]) if alt_list else 0

    # These exist in your TFExamples (from sanity check)
    variant_type = int(f["variant_type"].int64_list.value[0]) if "variant_type" in f else -1
    sequencing_type = int(f["sequencing_type"].int64_list.value[0]) if "sequencing_type" in f else -1
    vb = f["variant/encoded"].bytes_list.value[0] if "variant/encoded" in f else b""

    meta = {
        "alt_idx_list": alt_list_str,
        "alt_idx0": alt0,
        "chrom": chrom_l,
        "locus_start": s_l,
        "locus_end": e_l,
        "variant_type": variant_type,
        "sequencing_type": sequencing_type,
    }

    if emit_vcf_meta:
        meta.update(
            decode_variant_vcf_meta(
                variant_bytes=vb,
                alt_indices=alt_list,
                chrom_l=chrom_l,
                s_l=s_l,
                e_l=e_l,
                strict=strict_vcf_meta,
            )
        )    

    if emit_identity_meta:
        vhash = _sha1_hex(vb) if vb else ""
        vlen = len(vb)

        if meta.get("variant_key"):
            example_id = f"{meta['variant_key']}:alt_idx={alt_list_str}"
        elif vhash:
            example_id = f"{vhash}:{alt_list_str}"
        else:
            example_id = f"{chrom_l}:{s_l}-{e_l}:alt={alt_list_str}"

        meta.update({
            "variant_hash": vhash,
            "example_id": example_id,
            "variant_len": vlen,
        })
    return img, y, locus, meta


def _list_tensor_candidates(g: tf.Graph, last_dim: Optional[int] = None, contains: Optional[str] = None):
    cands = []
    for op in g.get_operations():
        for t in op.outputs:
            if not t.dtype.is_floating:
                continue
            shape = t.shape.as_list()
            if shape is None or len(shape) != 2:
                continue
            if last_dim is not None and shape[-1] != last_dim:
                continue
            name = t.name
            if contains is not None and contains not in name:
                continue
            cands.append(t)
    return cands


def _pick_embedding_tensor(g: tf.Graph) -> tf.Tensor:
    try:
        return g.get_tensor_by_name(DEFAULT_EMB_NAME)
    except Exception:
        pass

    cands = _list_tensor_candidates(g, last_dim=2048)
    if not cands:
        raise RuntimeError("Could not auto-detect embedding tensor. Pass --emb_tensor.")

    def score(t: tf.Tensor) -> int:
        n = t.name
        s = 0
        if "dropout" in n:
            s += 5
        if "inception" in n:
            s += 2
        if n.endswith("Identity:0"):
            s += 1
        return s

    cands.sort(key=score, reverse=True)
    return cands[0]


def _pick_logits_tensor(g: tf.Graph) -> tf.Tensor:
    try:
        return g.get_tensor_by_name(DEFAULT_LOGITS_NAME)
    except Exception:
        pass

    cands = _list_tensor_candidates(g, last_dim=3)
    if not cands:
        raise RuntimeError("Could not auto-detect logits tensor. Pass --logits_tensor.")

    def score(t: tf.Tensor) -> int:
        n = t.name
        s = 0
        if "classification" in n or "logits" in n:
            s += 5
        if t.op.type == "BiasAdd":
            s += 3
        if n.endswith("BiasAdd:0"):
            s += 1
        return s

    cands.sort(key=score, reverse=True)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="SavedModel dir (e.g. /opt/models/wgs)")
    ap.add_argument("--tfrecord_glob", required=True, help="Glob for sharded TFRecord(.gz)")
    ap.add_argument("--out_prefix", required=True, help="Output prefix for NPZ shards")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--chunk_size", type=int, default=40000)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=2000)

    ap.add_argument("--emb_tensor", default=None, help="Override embedding tensor name")
    ap.add_argument("--logits_tensor", default=None, help="Override logits tensor name")
    ap.add_argument("--verbose_tensors", action="store_true", help="Print candidate tensors and exit")

    ap.add_argument(
        "--emit_identity_meta",
        type=int,
        default=0,
        help="If 1, export legacy identity metadata: variant_hash, example_id, variant_len.",
    )
    ap.add_argument(
        "--emit_vcf_meta",
        type=int,
        default=1,
        help="If 1, export VCF-oriented metadata decoded from variant/encoded: "
             "vcf_chrom, vcf_pos, vcf_ref, vcf_alt, variant_key, etc.",
    )
    ap.add_argument(
        "--strict_vcf_meta",
        type=int,
        default=1,
        help="If 1, fail if VCF metadata cannot be decoded. If 0, emit empty fallback fields.",
    )    
    args = ap.parse_args()

    emit_identity_meta = bool(args.emit_identity_meta)
    emit_vcf_meta = bool(args.emit_vcf_meta)
    strict_vcf_meta = bool(args.strict_vcf_meta)    

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(args.tfrecord_glob))
    if not paths:
        raise SystemExit(f"No TFRecords matched: {args.tfrecord_glob}")

    print("TF version:", tf.__version__, "Eager:", tf.executing_eagerly())
    print("TFRecords:", len(paths))
    print("emit_identity_meta:", emit_identity_meta)
    print("emit_vcf_meta:", emit_vcf_meta)
    print("strict_vcf_meta:", strict_vcf_meta)

    loaded = tf.saved_model.load(args.model_dir)
    fn = loaded.signatures.get("serving_default")
    if fn is None:
        raise SystemExit("SavedModel missing 'serving_default' signature")

    frozen = convert_variables_to_constants_v2(fn)
    g = frozen.graph

    in_t = frozen.inputs[0]
    probs_t = frozen.outputs[0]  # safest: signature output

    if args.verbose_tensors:
        print("=== Candidates (None,2048) ===")
        for t in _list_tensor_candidates(g, last_dim=2048)[:50]:
            print(t.name, t.shape, "op=", t.op.type)
        print("=== Candidates (None,3) ===")
        for t in _list_tensor_candidates(g, last_dim=3)[:50]:
            print(t.name, t.shape, "op=", t.op.type)
        return

    emb_t = g.get_tensor_by_name(args.emb_tensor) if args.emb_tensor else _pick_embedding_tensor(g)
    logits_t = g.get_tensor_by_name(args.logits_tensor) if args.logits_tensor else _pick_logits_tensor(g)

    print("Input   :", in_t.name, in_t.shape)
    print("Emb     :", emb_t.name, emb_t.shape)
    print("Logits  :", logits_t.name, logits_t.shape, "op=", logits_t.op.type)
    print("Probs   :", probs_t.name, probs_t.shape, "op=", probs_t.op.type)

    pruned = frozen.prune(in_t, [emb_t, logits_t, probs_t])

    ds = tf.data.TFRecordDataset(paths, compression_type="GZIP")

    # Accumulators
    X: List[np.ndarray] = []
    Lg: List[np.ndarray] = []
    Pb: List[np.ndarray] = []
    Y: List[int] = []
    Loc: List[str] = []

    alt_idx_list: List[str] = []
    alt_idx0: List[int] = []

    chrom: List[str] = []
    locus_start: List[int] = []
    locus_end: List[int] = []
    variant_type: List[int] = []
    sequencing_type: List[int] = []

    if emit_vcf_meta:
        vcf_chrom: List[str] = []
        vcf_pos: List[int] = []
        vcf_start0: List[int] = []
        vcf_end0: List[int] = []
        vcf_ref: List[str] = []
        vcf_alt: List[str] = []
        vcf_alt_full: List[str] = []
        variant_key: List[str] = []
        variant_key_full: List[str] = []
        variant_encoded_len: List[int] = []

    if emit_identity_meta:
        variant_hash: List[str] = []
        example_id: List[str] = []
        variant_len: List[int] = []

    bx: List[np.ndarray] = []
    by: List[int] = []
    bloc: List[str] = []
    bmeta: List[dict] = []

    chunk = 0
    total = 0

    def flush():
        nonlocal chunk
        if not X:
            return

        out = f"{args.out_prefix}_{chunk:04d}.npz"

        payload = {
            "embeddings": np.stack(X).astype(np.float32),
            "teacher_logits": np.stack(Lg).astype(np.float32),
            "teacher_probs": np.stack(Pb).astype(np.float32),
            "label": np.asarray(Y, dtype=np.int64),
            "locus": np.asarray(Loc, dtype=np.str_),
            "alt_idx_list": np.asarray(alt_idx_list, dtype=np.str_),
            "alt_idx0": np.asarray(alt_idx0, dtype=np.int64),
            "chrom": np.asarray(chrom, dtype=np.str_),
            "locus_start": np.asarray(locus_start, dtype=np.int64),
            "locus_end": np.asarray(locus_end, dtype=np.int64),
            "variant_type": np.asarray(variant_type, dtype=np.int64),
            "sequencing_type": np.asarray(sequencing_type, dtype=np.int64),
        }

        if emit_vcf_meta:
            payload.update({
                "vcf_chrom": np.asarray(vcf_chrom, dtype=np.str_),
                "vcf_pos": np.asarray(vcf_pos, dtype=np.int64),
                "vcf_start0": np.asarray(vcf_start0, dtype=np.int64),
                "vcf_end0": np.asarray(vcf_end0, dtype=np.int64),
                "vcf_ref": np.asarray(vcf_ref, dtype=np.str_),
                "vcf_alt": np.asarray(vcf_alt, dtype=np.str_),
                "vcf_alt_full": np.asarray(vcf_alt_full, dtype=np.str_),
                "variant_key": np.asarray(variant_key, dtype=np.str_),
                "variant_key_full": np.asarray(variant_key_full, dtype=np.str_),
                "variant_encoded_len": np.asarray(variant_encoded_len, dtype=np.int64),
            })

        if emit_identity_meta:
            payload.update({
                "variant_hash": np.asarray(variant_hash, dtype=np.str_),
                "example_id": np.asarray(example_id, dtype=np.str_),
                "variant_len": np.asarray(variant_len, dtype=np.int64),
            })

        n_rows = len(X)

        expected_lengths = {
            "Y": len(Y),
            "Loc": len(Loc),
            "alt_idx_list": len(alt_idx_list),
            "chrom": len(chrom),
            "locus_start": len(locus_start),
            "variant_type": len(variant_type),
            "sequencing_type": len(sequencing_type),
            "alt_idx0": len(alt_idx0),
            "locus_end": len(locus_end),
        }

        if emit_vcf_meta:
            expected_lengths.update({
                "vcf_start0": len(vcf_start0),
                "vcf_end0": len(vcf_end0),
                "vcf_alt_full": len(vcf_alt_full),
                "variant_key_full": len(variant_key_full),
                "variant_encoded_len": len(variant_encoded_len),
            })

        if emit_identity_meta:
            expected_lengths.update({
                "variant_hash": len(variant_hash),
                "example_id": len(example_id),
                "variant_len": len(variant_len),
            })

        bad = {k: v for k, v in expected_lengths.items() if v != n_rows}
        if bad:
            raise RuntimeError(
                f"Accumulator length mismatch before saving {out}: "
                f"N={n_rows}, bad_lengths={bad}"
            )

        np.savez_compressed(out, **payload)

        print("Saved", out, "N=", len(X))
        chunk += 1

        X.clear(); Lg.clear(); Pb.clear(); Y.clear(); Loc.clear()
        alt_idx_list.clear(); alt_idx0.clear()
        chrom.clear(); locus_start.clear(); locus_end.clear()
        variant_type.clear(); sequencing_type.clear()

        if emit_vcf_meta:
            vcf_chrom.clear()
            vcf_pos.clear()
            vcf_start0.clear()
            vcf_end0.clear()
            vcf_ref.clear()
            vcf_alt.clear()
            vcf_alt_full.clear()
            variant_key.clear()
            variant_key_full.clear()
            variant_encoded_len.clear()

        if emit_identity_meta:
            variant_hash.clear(); example_id.clear(); variant_len.clear()

    def append_exported_row(emb_i, logits_i, probs_i, y_i, locus_i, m: dict) -> None:
        X.append(emb_i)
        Lg.append(logits_i)
        Pb.append(probs_i)
        Y.append(y_i)
        Loc.append(locus_i)

        alt_idx_list.append(m.get("alt_idx_list", ""))
        alt_idx0.append(int(m.get("alt_idx0", 0)))

        chrom.append(m.get("chrom", ""))
        locus_start.append(int(m.get("locus_start", -1)))
        locus_end.append(int(m.get("locus_end", -1)))
        variant_type.append(int(m.get("variant_type", -1)))
        sequencing_type.append(int(m.get("sequencing_type", -1)))

        if emit_vcf_meta:
            vcf_chrom.append(m.get("vcf_chrom", ""))
            vcf_pos.append(int(m.get("vcf_pos", -1)))
            vcf_start0.append(int(m.get("vcf_start0", -1)))
            vcf_end0.append(int(m.get("vcf_end0", -1)))
            vcf_ref.append(m.get("vcf_ref", ""))
            vcf_alt.append(m.get("vcf_alt", ""))
            vcf_alt_full.append(m.get("vcf_alt_full", ""))
            variant_key.append(m.get("variant_key", ""))
            variant_key_full.append(m.get("variant_key_full", ""))
            variant_encoded_len.append(int(m.get("variant_encoded_len", 0)))

        if emit_identity_meta:
            variant_hash.append(m.get("variant_hash", ""))
            example_id.append(m.get("example_id", ""))
            variant_len.append(int(m.get("variant_len", 0)))

    for rec in ds:
        img, y, locus, meta = parse_example(
            rec.numpy(),
            emit_identity_meta=emit_identity_meta,
            emit_vcf_meta=emit_vcf_meta,
            strict_vcf_meta=strict_vcf_meta,
        )
        bx.append(img); by.append(y); bloc.append(locus); bmeta.append(meta)

        if len(bx) >= args.batch_size:
            batch = tf.convert_to_tensor(np.stack(bx).astype(np.float32))
            emb, logits, probs = pruned(batch)
            emb = emb.numpy(); logits = logits.numpy(); probs = probs.numpy()

            if total == 0:
                s = float(np.mean(np.sum(probs, axis=1)))
                print("Sanity mean(sum(probs)) =", s)
                try:
                    e = np.max(np.abs(tf.nn.softmax(logits, axis=1).numpy() - probs))
                    print("Sanity max|softmax(logits)-probs| =", float(e))
                except Exception as ex:
                    print("Sanity softmax check skipped:", ex)

            for i in range(emb.shape[0]):
                append_exported_row(
                    emb_i=emb[i],
                    logits_i=logits[i],
                    probs_i=probs[i],
                    y_i=by[i],
                    locus_i=bloc[i],
                    m=bmeta[i],
                )

            total += emb.shape[0]
            if total % args.log_every == 0:
                print("Processed:", total)

            bx.clear(); by.clear(); bloc.clear(); bmeta.clear()
            if len(X) >= args.chunk_size:
                flush()

        if args.max_records is not None and total >= args.max_records:
            break

    # final partial batch
    if bx and (args.max_records is None or total < args.max_records):
        batch = tf.convert_to_tensor(np.stack(bx).astype(np.float32))
        emb, logits, probs = pruned(batch)
        emb = emb.numpy(); logits = logits.numpy(); probs = probs.numpy()

        for i in range(emb.shape[0]):
            append_exported_row(
                emb_i=emb[i],
                logits_i=logits[i],
                probs_i=probs[i],
                y_i=by[i],
                locus_i=bloc[i],
                m=bmeta[i],
            )

        total += emb.shape[0]

    flush()
    print("Done. Total:", total)


if __name__ == "__main__":
    main()