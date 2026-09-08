"""
Shared confusion-matrix metrics used by both canonical trainers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def cm_3(y_true: List[int], y_pred: List[int], n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm_3(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    n_classes = cm.shape[0]
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))

    per_class = {}
    f1s = []
    recs = []
    precs = []
    supports = []

    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        support = float(cm[c, :].sum())

        prec = float(tp / (tp + fp + eps))
        rec = float(tp / (tp + fn + eps))
        f1 = float(2 * prec * rec / (prec + rec + eps))

        per_class[str(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support),
        }

        f1s.append(f1)
        recs.append(rec)
        precs.append(prec)
        supports.append(support)

    macro_f1 = float(np.mean(f1s))
    macro_recall = float(np.mean(recs))
    macro_precision = float(np.mean(precs))
    wsum = float(np.sum(supports)) + eps
    weighted_f1 = float(np.sum([f1s[i] * supports[i] for i in range(n_classes)]) / wsum)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }
