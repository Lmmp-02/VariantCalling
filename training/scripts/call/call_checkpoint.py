#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run trained checkpoints in calling mode and generate an intermediate calls CSV.

Purpose
-------
This script is NOT an evaluator. It does not require labels and does not compute
supervised metrics.

It loads a trained checkpoint and applies it to calling-mode offline datasets:

1. Unimodal checkpoints:
   - Illumina model -> data/4_out/datasets/unimodal/<dataset_id>/illumina_wgs/calling/
   - ONT model      -> data/4_out/datasets/unimodal/<dataset_id>/ont_r104/calling/

2. Hybrid checkpoints:
   - Hybrid model -> data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/

Output
------
A clean intermediate CSV with:
- VCF-ready candidate metadata already decoded upstream
- model prediction
- prediction probabilities/logits
- traceability fields

This script writes all rows, including 0/0 predictions. It does not decide
FILTER, QUAL, RefCall, PASS, no-call, or whether a row should be emitted in the
final VCF. Those policies belong to a later calls_to_vcf.py step.

The CSV is intended as the input to a future calls_to_vcf.py script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Make repo root importable when executing as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models import (  # noqa: E402
    GroupwiseLinear3C,
    GroupwiseMLP3C,
    LinearHead,
    MLP,
)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------

def build_model(
    *,
    model_kind: str,
    model_family: Optional[str],
    in_dim: int,
    n_classes: int,
    arch: str,
    hidden: int,
    dropout: float,
) -> nn.Module:
    if model_kind == "unimodal":
        if arch == "linear":
            return LinearHead(in_dim=in_dim, n_classes=n_classes)
        return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    if model_kind == "hybrid":
        if model_family == "hybrid_simple_3c":
            if arch == "linear":
                return LinearHead(in_dim=in_dim, n_classes=n_classes)
            return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

        if model_family == "hybrid_groupwise_3c":
            if arch == "linear":
                return GroupwiseLinear3C(in_dim=in_dim, n_classes=n_classes)
            return GroupwiseMLP3C(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    raise SystemExit(f"[ERROR] Unsupported model kind/family: {model_kind} / {model_family}")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in {path}: {e}")


def torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def normalize_modality(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None

    s = str(x).lower()

    if s in {"ill", "illumina", "illumina_wgs"}:
        return "illumina"
    if s in {"ont", "ont_r104"}:
        return "ont"

    return s


def resolve_add_masks(ckpt: Dict[str, Any], config: Dict[str, Any]) -> bool:
    resolved = config.get("resolved", {})
    ckpt_args = ckpt.get("args", {}) or {}

    if "add_masks" in ckpt:
        return bool(ckpt["add_masks"])
    if "add_masks" in resolved:
        return bool(resolved["add_masks"])
    if "no_add_masks" in ckpt_args and bool(ckpt_args["no_add_masks"]):
        return False
    if "add_masks" in ckpt_args:
        return bool(ckpt_args["add_masks"])

    # Canonical default for current hybrid trainers.
    return True


def infer_checkpoint_info(
    *,
    ckpt: Dict[str, Any],
    config: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    resolved = config.get("resolved", {})
    model_policy = config.get("model_policy", {})
    ckpt_args = ckpt.get("args", {}) or {}

    modality = (
        ckpt.get("modality")
        or resolved.get("modality")
        or ckpt_args.get("modality")
    )
    modality = normalize_modality(modality)

    model_family = (
        ckpt.get("model_family")
        or resolved.get("model_family")
        or model_policy.get("model_family")
        or ckpt_args.get("model_family")
    )

    architecture = (
        ckpt.get("architecture")
        or resolved.get("architecture")
        or model_policy.get("architecture")
        or ckpt_args.get("architecture")
    )

    if model_family is None and architecture is not None:
        architecture = str(architecture)
        if architecture == "groupwise_3class":
            model_family = "hybrid_groupwise_3c"
        elif architecture in {"shared_3class", "hybrid_simple_3c"}:
            model_family = "hybrid_simple_3c"

    # Last-resort inference from experiment name.
    exp_lower = experiment_name.lower()
    if modality is None and model_family is None:
        if "unimodal" in exp_lower and "illumina" in exp_lower:
            modality = "illumina"
        elif "unimodal" in exp_lower and "ont" in exp_lower:
            modality = "ont"
        elif "hybrid_groupwise" in exp_lower:
            model_family = "hybrid_groupwise_3c"
        elif "hybrid_simple" in exp_lower or "hybrid" in exp_lower:
            model_family = "hybrid_simple_3c"

    if modality is not None:
        model_kind = "unimodal"
        model_family = None
    elif model_family is not None:
        model_kind = "hybrid"
        model_family = str(model_family)
    else:
        raise SystemExit(
            "[ERROR] Cannot infer checkpoint type. "
            "Expected modality for unimodal or model_family for hybrid."
        )

    input_dim = int(
        ckpt.get("input_dim")
        or resolved.get("input_dim")
        or ckpt_args.get("input_dim")
        or 0
    )

    n_classes = int(
        ckpt.get("n_classes")
        or resolved.get("n_classes")
        or ckpt_args.get("n_classes")
        or 3
    )

    arch = str(
        ckpt.get("arch")
        or resolved.get("arch")
        or ckpt_args.get("arch")
        or "mlp"
    ).lower()

    hidden = int(
        ckpt.get("hidden")
        or resolved.get("hidden")
        or ckpt_args.get("hidden")
        or 512
    )

    dropout = float(
        ckpt.get("dropout")
        or resolved.get("dropout")
        or ckpt_args.get("dropout")
        or 0.1
    )

    add_masks = resolve_add_masks(ckpt, config)

    return {
        "model_kind": model_kind,
        "modality": modality,
        "model_family": model_family,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "arch": arch,
        "hidden": hidden,
        "dropout": dropout,
        "add_masks": add_masks,
    }


def list_npz_paths(data_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"[ERROR] No NPZ shards found in {data_dir}")
    return paths


def get_str_array(d: np.lib.npyio.NpzFile, key: str, n: int, default: str = "") -> np.ndarray:
    if key in d.files:
        return d[key].astype(str)
    return np.full((n,), default, dtype=object)


def get_int_array(d: np.lib.npyio.NpzFile, key: str, n: int, default: int = -1) -> np.ndarray:
    if key in d.files:
        return d[key].astype(np.int64)
    return np.full((n,), int(default), dtype=np.int64)


def gt_from_pred(pred_label: int) -> str:
    if int(pred_label) == 0:
        return "0/0"
    if int(pred_label) == 1:
        return "0/1"
    if int(pred_label) == 2:
        return "1/1"
    return "./."


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def infer_input_dim_from_first_shard(
    *,
    input_kind: str,
    npz_paths: List[Path],
    add_masks: bool,
) -> int:
    p = npz_paths[0]
    d = np.load(p, allow_pickle=True)
    try:
        if input_kind == "unimodal":
            return int(d["embeddings"].shape[1])

        in_dim = int(d["ill_embeddings"].shape[1] + d["ont_embeddings"].shape[1])
        if add_masks:
            in_dim += 2
        return in_dim
    finally:
        d.close()


# ---------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------

CSV_FIELDS = [
    "dataset_id",
    "model_name",
    "input_kind",

    "shard",
    "row_idx",

    "chrom",
    "pos",
    "ref",
    "alt",
    "format",
    "gt",

    "pred_label",
    "pred_logit_0",
    "pred_logit_1",
    "pred_logit_2",
    "pred_prob_0",
    "pred_prob_1",
    "pred_prob_2",

    "locus",
    "variant_key",
    "variant_key_full",
    "vcf_source",
    "vcf_key_mismatch",

    "group",
    "mask_ill",
    "mask_ont",
    "variant_type",
]


def make_output_row(
    *,
    dataset_id: str,
    input_kind: str,
    model_name: str,
    shard_path: Path,
    shard_row: int,
    locus: str,
    group: int,
    mask_ill: int,
    mask_ont: int,
    variant_type: int,
    vcf_chrom: str,
    vcf_pos: int,
    vcf_ref: str,
    vcf_alt: str,
    variant_key: str,
    variant_key_full: str,
    vcf_source: str,
    vcf_key_mismatch: int,
    logits: np.ndarray,
    probs: np.ndarray,
    pred_label: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pred_gt = gt_from_pred(pred_label)

    missing_vcf_meta = int(
        not str(vcf_chrom).strip()
        or int(vcf_pos) < 1
        or not str(vcf_ref).strip()
        or not str(vcf_alt).strip()
    )

    row = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "input_kind": input_kind,

        "shard": str(shard_path),
        "row_idx": int(shard_row),

        "chrom": str(vcf_chrom),
        "pos": int(vcf_pos),
        "ref": str(vcf_ref),
        "alt": str(vcf_alt),
        "format": "GT",
        "gt": pred_gt,

        "pred_label": int(pred_label),
        "pred_logit_0": float(logits[0]),
        "pred_logit_1": float(logits[1]),
        "pred_logit_2": float(logits[2]),
        "pred_prob_0": float(probs[0]),
        "pred_prob_1": float(probs[1]),
        "pred_prob_2": float(probs[2]),

        "locus": str(locus),
        "variant_key": str(variant_key),
        "variant_key_full": str(variant_key_full),
        "vcf_source": str(vcf_source),
        "vcf_key_mismatch": int(vcf_key_mismatch),

        "group": int(group),
        "mask_ill": int(mask_ill),
        "mask_ont": int(mask_ont),
        "variant_type": int(variant_type),
    }

    stat_update = {
        "pred_label": int(pred_label),
        "gt": pred_gt,
        "vcf_key_mismatch": int(vcf_key_mismatch),
        "missing_vcf_meta": missing_vcf_meta,
        "group": int(group),
        "variant_type": int(variant_type),
    }

    return row, stat_update


def update_stats(stats: Dict[str, Any], upd: Dict[str, Any]) -> None:
    stats["total_rows"] += 1
    stats["pred_counts"][str(upd["pred_label"])] += 1
    stats["gt_counts"][str(upd["gt"])] += 1
    stats["group_counts"][str(upd["group"])] += 1
    stats["variant_type_counts"][str(upd["variant_type"])] += 1

    if int(upd["vcf_key_mismatch"]) == 1:
        stats["vcf_key_mismatch_rows"] += 1

    if int(upd["missing_vcf_meta"]) == 1:
        stats["missing_vcf_meta_rows"] += 1


# ---------------------------------------------------------------------
# Calling loops
# ---------------------------------------------------------------------

@torch.no_grad()
def call_unimodal(
    *,
    model: nn.Module,
    npz_paths: List[Path],
    modality: str,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    info: Dict[str, Any],
    writer: csv.DictWriter,
    stats: Dict[str, Any],
) -> None:
    model.eval()

    group_default = 10 if modality == "illumina" else 1
    mask_ill_default = 1 if modality == "illumina" else 0
    mask_ont_default = 1 if modality == "ont" else 0

    for shard_path in npz_paths:
        d = np.load(shard_path, allow_pickle=True)
        try:
            emb = d["embeddings"].astype(np.float32, copy=False)
            n = int(emb.shape[0])

            if int(emb.shape[1]) != int(info["input_dim"]):
                raise SystemExit(
                    f"[ERROR] Input dim mismatch in {shard_path}: "
                    f"features={emb.shape[1]} checkpoint_input_dim={info['input_dim']}"
                )

            locus = get_str_array(d, "locus", n)
            variant_type = get_int_array(d, "variant_type", n, -1)

            vcf_chrom = get_str_array(d, "vcf_chrom", n)
            vcf_pos = get_int_array(d, "vcf_pos", n, -1)
            vcf_ref = get_str_array(d, "vcf_ref", n)
            vcf_alt = get_str_array(d, "vcf_alt", n)
            variant_key = get_str_array(d, "variant_key", n)
            variant_key_full = get_str_array(d, "variant_key_full", n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x = torch.from_numpy(emb[start:end]).to(device)

                logits_t = model(x)
                probs_t = softmax_torch(logits_t)
                pred_t = torch.argmax(logits_t, dim=1)

                logits_np = logits_t.detach().cpu().numpy()
                probs_np = probs_t.detach().cpu().numpy()
                pred_np = pred_t.detach().cpu().numpy().astype(np.int64)

                for j in range(start, end):
                    local = j - start
                    pred_label = int(pred_np[local])

                    row, upd = make_output_row(
                        dataset_id=args.dataset_id,
                        input_kind="unimodal",
                        model_name=args.experiment_dir.name,
                        shard_path=shard_path,
                        shard_row=j,
                        locus=str(locus[j]),
                        group=group_default,
                        mask_ill=mask_ill_default,
                        mask_ont=mask_ont_default,
                        variant_type=int(variant_type[j]),
                        vcf_chrom=str(vcf_chrom[j]),
                        vcf_pos=int(vcf_pos[j]),
                        vcf_ref=str(vcf_ref[j]),
                        vcf_alt=str(vcf_alt[j]),
                        variant_key=str(variant_key[j]),
                        variant_key_full=str(variant_key_full[j]),
                        vcf_source=modality,
                        vcf_key_mismatch=0,
                        logits=logits_np[local],
                        probs=probs_np[local],
                        pred_label=pred_label,
                    )

                    writer.writerow(row)
                    update_stats(stats, upd)

                    if args.limit_rows and stats["total_rows"] >= args.limit_rows:
                        return

        finally:
            d.close()


@torch.no_grad()
def call_hybrid(
    *,
    model: nn.Module,
    npz_paths: List[Path],
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    info: Dict[str, Any],
    writer: csv.DictWriter,
    stats: Dict[str, Any],
) -> None:
    model.eval()

    add_masks = bool(info["add_masks"])
    model_family = str(info["model_family"])

    for shard_path in npz_paths:
        d = np.load(shard_path, allow_pickle=True)
        try:
            ill_emb = d["ill_embeddings"].astype(np.float32, copy=False)
            ont_emb = d["ont_embeddings"].astype(np.float32, copy=False)
            n = int(ill_emb.shape[0])

            mask_ill = get_int_array(d, "mask_ill", n, 0)
            mask_ont = get_int_array(d, "mask_ont", n, 0)
            group = get_int_array(d, "group", n, -1)

            current_dim = int(ill_emb.shape[1] + ont_emb.shape[1] + (2 if add_masks else 0))
            if current_dim != int(info["input_dim"]):
                raise SystemExit(
                    f"[ERROR] Input dim mismatch in {shard_path}: "
                    f"features={current_dim} checkpoint_input_dim={info['input_dim']}"
                )

            locus = get_str_array(d, "locus", n)
            variant_type = get_int_array(d, "variant_type", n, -1)

            vcf_chrom = get_str_array(d, "vcf_chrom", n)
            vcf_pos = get_int_array(d, "vcf_pos", n, -1)
            vcf_ref = get_str_array(d, "vcf_ref", n)
            vcf_alt = get_str_array(d, "vcf_alt", n)
            variant_key = get_str_array(d, "variant_key", n)
            variant_key_full = get_str_array(d, "variant_key_full", n)
            vcf_source = get_str_array(d, "vcf_source", n)
            vcf_key_mismatch = get_int_array(d, "vcf_key_mismatch", n, 0)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                feats = [ill_emb[start:end], ont_emb[start:end]]
                if add_masks:
                    masks = np.stack(
                        [
                            mask_ill[start:end].astype(np.float32),
                            mask_ont[start:end].astype(np.float32),
                        ],
                        axis=1,
                    )
                    feats.append(masks)

                x_np = np.concatenate(feats, axis=1).astype(np.float32, copy=False)
                x = torch.from_numpy(x_np).to(device)
                g = torch.from_numpy(group[start:end].astype(np.int64)).to(device)

                if model_family == "hybrid_simple_3c":
                    logits_t = model(x)
                elif model_family == "hybrid_groupwise_3c":
                    logits_t = model(x, g)
                else:
                    raise SystemExit(f"[ERROR] Unsupported hybrid model_family={model_family}")

                probs_t = softmax_torch(logits_t)
                pred_t = torch.argmax(logits_t, dim=1)

                logits_np = logits_t.detach().cpu().numpy()
                probs_np = probs_t.detach().cpu().numpy()
                pred_np = pred_t.detach().cpu().numpy().astype(np.int64)

                for j in range(start, end):
                    local = j - start
                    pred_label = int(pred_np[local])

                    row, upd = make_output_row(
                        dataset_id=args.dataset_id,
                        input_kind="multimodal_outer",
                        model_name=args.experiment_dir.name,
                        shard_path=shard_path,
                        shard_row=j,
                        locus=str(locus[j]),
                        group=int(group[j]),
                        mask_ill=int(mask_ill[j]),
                        mask_ont=int(mask_ont[j]),
                        variant_type=int(variant_type[j]),
                        vcf_chrom=str(vcf_chrom[j]),
                        vcf_pos=int(vcf_pos[j]),
                        vcf_ref=str(vcf_ref[j]),
                        vcf_alt=str(vcf_alt[j]),
                        variant_key=str(variant_key[j]),
                        variant_key_full=str(variant_key_full[j]),
                        vcf_source=str(vcf_source[j]),
                        vcf_key_mismatch=int(vcf_key_mismatch[j]),
                        logits=logits_np[local],
                        probs=probs_np[local],
                        pred_label=pred_label,
                    )

                    writer.writerow(row)
                    update_stats(stats, upd)

                    if args.limit_rows and stats["total_rows"] >= args.limit_rows:
                        return

        finally:
            d.close()


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

def resolve_input_kind(args: argparse.Namespace, info: Dict[str, Any]) -> str:
    if args.input_kind != "auto":
        return args.input_kind

    if info["model_kind"] == "unimodal":
        return "unimodal"
    if info["model_kind"] == "hybrid":
        return "multimodal_outer"

    raise SystemExit(f"[ERROR] Cannot resolve input kind for model_kind={info['model_kind']}")


def resolve_npz_paths(args: argparse.Namespace, info: Dict[str, Any], input_kind: str) -> List[Path]:
    if input_kind == "unimodal":
        modality = info.get("modality")
        if modality not in {"illumina", "ont"}:
            raise SystemExit(f"[ERROR] Unimodal checkpoint has invalid modality={modality}")

        if args.unimodal_dir:
            data_dir = Path(args.unimodal_dir).resolve()
        else:
            preset = "illumina_wgs" if modality == "illumina" else "ont_r104"
            data_dir = (
                Path(args.unimodal_root).resolve()
                / args.dataset_id
                / preset
                / args.mode
            )

        preset = "illumina_wgs" if modality == "illumina" else "ont_r104"
        pattern = f"{args.dataset_id}.{preset}.{args.mode}_*.npz"
        return list_npz_paths(data_dir, pattern)

    if input_kind == "multimodal_outer":
        if args.multimodal_outer_dir:
            data_dir = Path(args.multimodal_outer_dir).resolve()
        else:
            data_dir = (
                Path(args.multimodal_root).resolve()
                / args.dataset_id
                / "outer"
            )

        pattern = f"{args.dataset_id}_outer_*.npz"
        return list_npz_paths(data_dir, pattern)

    raise SystemExit(f"[ERROR] Unsupported input_kind={input_kind}")


def default_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path("training/out/calls").resolve() / args.experiment_dir.name / args.dataset_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / "calls.csv", out_dir / "call_report.json"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--experiment_dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--dataset_id", type=str, required=True)

    ap.add_argument("--mode", choices=["calling"], default="calling")
    ap.add_argument("--input_kind", choices=["auto", "unimodal", "multimodal_outer"], default="auto")

    ap.add_argument("--unimodal_root", type=str, default="data/4_out/datasets/unimodal")
    ap.add_argument("--multimodal_root", type=str, default="data/4_out/datasets/multimodal/by_subject")
    ap.add_argument("--unimodal_dir", type=str, default=None)
    ap.add_argument("--multimodal_outer_dir", type=str, default=None)

    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--limit_rows", type=int, default=0, help="Optional smoke-test row cap. 0 = no cap.")

    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_report", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)

    args = ap.parse_args()

    args.experiment_dir = args.experiment_dir.resolve()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[ERROR] --device cuda requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.experiment_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    config_path = args.experiment_dir / "config.json"
    config = load_json(config_path)
    ckpt = torch_load(ckpt_path)

    info = infer_checkpoint_info(
        ckpt=ckpt,
        config=config,
        experiment_name=args.experiment_dir.name,
    )

    input_kind = resolve_input_kind(args, info)
    npz_paths = resolve_npz_paths(args, info, input_kind)

    if info["input_dim"] <= 0:
        info["input_dim"] = infer_input_dim_from_first_shard(
            input_kind=input_kind,
            npz_paths=npz_paths,
            add_masks=bool(info["add_masks"]),
        )

    model = build_model(
        model_kind=info["model_kind"],
        model_family=info.get("model_family"),
        in_dim=int(info["input_dim"]),
        n_classes=int(info["n_classes"]),
        arch=str(info["arch"]),
        hidden=int(info["hidden"]),
        dropout=float(info["dropout"]),
    )

    if "model_state" not in ckpt:
        raise SystemExit("[ERROR] Checkpoint does not contain model_state.")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    default_csv, default_report = default_output_paths(args)
    out_csv = Path(args.out_csv).resolve() if args.out_csv else default_csv
    out_report = Path(args.out_report).resolve() if args.out_report else default_report

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "total_rows": 0,
        "vcf_key_mismatch_rows": 0,
        "missing_vcf_meta_rows": 0,
        "pred_counts": Counter(),
        "gt_counts": Counter(),
        "group_counts": Counter(),
        "variant_type_counts": Counter(),
    }

    print("=== CALL CHECKPOINT ===")
    print(f"experiment_dir       : {args.experiment_dir}")
    print(f"checkpoint           : {ckpt_path}")
    print(f"dataset_id           : {args.dataset_id}")
    print(f"input_kind           : {input_kind}")
    print(f"n_npz_shards         : {len(npz_paths)}")
    print(f"first_npz            : {npz_paths[0]}")
    print(f"device               : {device}")
    print(f"batch_size           : {args.batch_size}")
    print(f"limit_rows           : {args.limit_rows}")
    print(f"model_info           : {json.dumps(_jsonify(info), indent=2)}")
    print("=======================")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        if input_kind == "unimodal":
            call_unimodal(
                model=model,
                npz_paths=npz_paths,
                modality=str(info["modality"]),
                device=device,
                batch_size=args.batch_size,
                args=args,
                info=info,
                writer=writer,
                stats=stats,
            )
        elif input_kind == "multimodal_outer":
            call_hybrid(
                model=model,
                npz_paths=npz_paths,
                device=device,
                batch_size=args.batch_size,
                args=args,
                info=info,
                writer=writer,
                stats=stats,
            )
        else:
            raise SystemExit(f"[ERROR] Unsupported input_kind={input_kind}")

    report = {
        "experiment_dir": str(args.experiment_dir),
        "checkpoint": str(ckpt_path),
        "dataset_id": args.dataset_id,
        "input_kind": input_kind,
        "npz_shards": [str(p) for p in npz_paths],
        "device": str(device),
        "batch_size": args.batch_size,
        "limit_rows": args.limit_rows,
        "model_info": info,
        "out_csv": str(out_csv),
        "stats": stats,
    }

    out_report.write_text(json.dumps(_jsonify(report), indent=2, sort_keys=True), encoding="utf-8")

    print("\n[Saved]")
    print(f"CSV calls report : {out_csv}")
    print(f"JSON report      : {out_report}")

    print("\n[Summary]")
    print(f"total_rows            : {stats['total_rows']:,}")
    print(f"vcf_key_mismatch_rows : {stats['vcf_key_mismatch_rows']:,}")
    print(f"missing_vcf_meta_rows : {stats['missing_vcf_meta_rows']:,}")
    print(f"pred_counts           : {dict(stats['pred_counts'])}")
    print(f"gt_counts             : {dict(stats['gt_counts'])}")
    print(f"group_counts          : {dict(stats['group_counts'])}")
    print(f"variant_type_counts   : {dict(stats['variant_type_counts'])}")


if __name__ == "__main__":
    main()
