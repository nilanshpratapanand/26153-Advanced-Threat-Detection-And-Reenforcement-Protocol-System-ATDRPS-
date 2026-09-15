"""Metrics and the benchmark table.

Everything the problem statement asks to be measured is measured here, on
identical test splits and identical features for every model:

* **Infiltration forecasting** -- F1, precision, recall and false-positive rate
  at each horizon step, plus ROC-AUC and average precision.
* **MITRE stage forecasting** -- accuracy, macro-F1, per-class scores and the
  confusion matrix.
* **Dynamics** -- next-state error against the persistence floor.  This is the
  one that decides whether the thing is a world model at all.  A model can win
  on stage F1 while being unable to predict the next state better than "assume
  nothing changes", and that model has learned a classifier, not dynamics.

False-positive rate is reported rather than accuracy alone because a SOC lives
or dies by it: at these class balances a model that predicts "benign" for
everything scores about 0.8 accuracy and is worthless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BinaryMetrics", "StageMetrics", "binary_metrics", "stage_metrics",
    "dynamics_metrics", "benchmark_table", "find_best_threshold",
]


@dataclass
class BinaryMetrics:
    f1: float
    precision: float
    recall: float
    fpr: float
    accuracy: float
    roc_auc: float
    average_precision: float
    tp: int
    fp: int
    tn: int
    fn: int
    threshold: float = 0.5
    support_positive: int = 0

    def as_row(self) -> dict:
        return {
            "F1": self.f1, "Precision": self.precision, "Recall": self.recall,
            "FPR": self.fpr, "Accuracy": self.accuracy, "ROC-AUC": self.roc_auc,
            "AP": self.average_precision,
        }


@dataclass
class StageMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict = field(default_factory=dict)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    labels: list[str] = field(default_factory=list)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray,
                   threshold: float = 0.5) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    pred = (scores >= threshold).astype(int)

    tp = int(np.count_nonzero((pred == 1) & (y_true == 1)))
    fp = int(np.count_nonzero((pred == 1) & (y_true == 0)))
    tn = int(np.count_nonzero((pred == 0) & (y_true == 0)))
    fn = int(np.count_nonzero((pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    roc = ap = float("nan")
    if np.unique(y_true).size == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        roc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))

    return BinaryMetrics(
        f1=f1, precision=precision, recall=recall, fpr=fpr, accuracy=accuracy,
        roc_auc=roc, average_precision=ap, tp=tp, fp=fp, tn=tn, fn=fn,
        threshold=float(threshold), support_positive=int((y_true == 1).sum()),
    )


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray,
                        metric: str = "f1", max_fpr: float | None = None) -> float:
    """Pick an operating point on the validation split.

    Never on test -- choosing a threshold on the data you then report is how
    benchmark numbers become fiction.  ``max_fpr`` lets a deployment pin the
    false-positive budget first and take the best recall available underneath
    it, which is how a SOC would actually tune this.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return 0.5
    candidates = np.unique(np.round(scores, 4))
    if candidates.size > 400:
        candidates = np.quantile(scores, np.linspace(0, 1, 400))
    best_value, best_threshold = -1.0, 0.5
    for threshold in candidates:
        m = binary_metrics(y_true, scores, float(threshold))
        if max_fpr is not None and m.fpr > max_fpr:
            continue
        value = {"f1": m.f1, "recall": m.recall, "precision": m.precision}[metric]
        if value > best_value:
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def stage_metrics(y_true: np.ndarray, probs: np.ndarray,
                  labels: list[str]) -> StageMetrics:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    y_true = np.asarray(y_true).astype(int).ravel()
    pred = np.asarray(probs).reshape(len(y_true), -1).argmax(axis=1)
    if y_true.size == 0:
        return StageMetrics(0.0, 0.0, 0.0, {}, np.zeros((0, 0)), labels)

    present = sorted(set(y_true.tolist()) | set(pred.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=present, zero_division=0
    )
    per_class = {
        labels[c] if c < len(labels) else str(c): {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i, c in enumerate(present)
    }
    return StageMetrics(
        accuracy=float((pred == y_true).mean()),
        macro_f1=float(f1_score(y_true, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion=confusion_matrix(y_true, pred, labels=present),
        labels=[labels[c] if c < len(labels) else str(c) for c in present],
    )


def dynamics_metrics(pred_state: np.ndarray, true_state: np.ndarray,
                     standardiser=None) -> dict:
    """Next-state error, reported in standardised units so features with wildly
    different scales contribute comparably."""
    pred = np.asarray(pred_state, dtype=np.float64)
    true = np.asarray(true_state, dtype=np.float64)
    if standardiser is not None:
        pred = standardiser.transform(pred.astype(np.float32)).astype(np.float64)
        true = standardiser.transform(true.astype(np.float32)).astype(np.float64)
    diff = pred - true
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def benchmark_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render a markdown table.  Used for docs/BENCHMARKS.md and the CLI."""
    if not rows:
        return "_no results_"
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in columns}
    head = "| " + " | ".join(str(c).ljust(widths[c]) for c in columns) + " |"
    rule = "|" + "|".join("-" * (widths[c] + 2) for c in columns) + "|"
    body = [
        "| " + " | ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in columns) + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body])


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value:          # NaN
            return "n/a"
        return f"{value:.4f}"
    return str(value)
