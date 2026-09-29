"""The onset-forecasting task and how it is scored.

Definitions (all in units of windows; ``W`` is the window length in seconds):

* A window is **active** if the ground truth marks infiltration in it.
* An **onset** is an active window whose predecessor is not active.
* A sample at time ``t`` is **quiet** if none of the last ``quiet_gap`` windows
  (including ``t``) is active.  Only quiet samples are forecasting problems: on a
  non-quiet sample the attack is already visible and "forecasting" it is detection.
* The label is ``1`` iff an onset occurs within ``t+1 .. t+K``.

What is reported, and why (Arp et al. 2022, Axelsson 2000, Pendlebury et al. 2019):

* AUC / average precision **on quiet samples only**, with cluster-bootstrap intervals
  (samples from one capture are correlated, so captures are resampled, not samples).
* TPR at fixed false-positive-rate budgets, threshold chosen on validation data and
  applied unchanged to test.
* Event-level recall and **warning lead time**: of the onsets that happened, how many
  were flagged before they began, and how early.
* Precision at an assumed *real* prevalence (base-rate correction), because precision
  measured on a 5 %-positive test set says nothing about a network where onsets are
  one in ten thousand windows.
* When the negative set is too small to resolve an FPR budget, the result says so
  instead of reporting an over-confident zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["OnsetSamples", "build_onset_samples", "split_groups", "evaluate", "contexts_from_states", "LATE_STAGES",
           "threshold_for_fpr", "PREVALENCES", "FPR_BUDGETS"]

PREVALENCES = (1e-2, 1e-3, 1e-4)      # assumed real onset rate per window
FPR_BUDGETS = (1e-2, 1e-3)


@dataclass
class OnsetSamples:
    context: np.ndarray        # (N, L, F) float32 -- windows t-L+1 .. t, nothing later
    onset_in_k: np.ndarray     # (N,) int8    -- 1 if an onset occurs in t+1 .. t+K
    steps_to_onset: np.ndarray  # (N,) int16  -- 1..K, or 0 when there is none
    group: np.ndarray          # (N,) int32   -- capture id
    t: np.ndarray              # (N,) int32   -- window index inside the capture
    onset_id: np.ndarray       # (N,) int64   -- id of the onset this sample precedes, -1 if none
    window_size_s: float
    horizon: int
    quiet_gap: int
    feature_names: list[str]

    def __len__(self) -> int:
        return int(self.context.shape[0])

    def subset(self, idx) -> "OnsetSamples":
        idx = np.asarray(idx)
        return replace(self, context=self.context[idx], onset_in_k=self.onset_in_k[idx],
                       steps_to_onset=self.steps_to_onset[idx], group=self.group[idx],
                       t=self.t[idx], onset_id=self.onset_id[idx])

    @property
    def n_positive(self) -> int:
        return int(self.onset_in_k.sum())

    @property
    def n_events(self) -> int:
        ids = self.onset_id[self.onset_id >= 0]
        return int(len(np.unique(ids)))

    def describe(self) -> str:
        n = len(self)
        return (f"{n} quiet samples from {len(np.unique(self.group))} captures, "
                f"{self.n_positive} positive ({100 * self.n_positive / max(n, 1):.2f}%), "
                f"{self.n_events} distinct onsets, horizon K={self.horizon}, "
                f"quiet gap {self.quiet_gap}")


LATE_STAGES = ("LateralMovement", "CommandAndControl", "Exfiltration", "Impact")


def build_onset_samples(captures, context: int = 16, horizon: int = 5,
                        quiet_gap: int = 3, group_offset: int = 0,
                        positive_stages=None) -> OnsetSamples:
    """Turn labelled, windowed captures into quiet-state forecasting samples.

    ``captures`` are :class:`~atdrps.data.windows.WindowedStates`.  A sample is dropped if
    any label it depends on is marked invalid, so unlabelled stretches never become
    silent negatives.

    ``positive_stages`` chooses what counts as "active".  ``None`` uses the capture's
    infiltration flag (Initial Access onwards: the *from-silence onset* task).  Passing
    ``LATE_STAGES`` defines the *escalation* task instead: reconnaissance and initial access
    stay observable precursors, and the question becomes whether compromise spreads
    (lateral movement, command-and-control, exfiltration, impact) within the horizon.
    """
    from ..data.schema import STAGES
    late_idx = None if positive_stages is None else [STAGES.index(x) for x in positive_stages]
    if context < 1 or horizon < 1 or quiet_gap < 1:
        raise ValueError("context, horizon and quiet_gap must all be >= 1")
    ctx, y, steps, grp, tt, oid = [], [], [], [], [], []
    feature_names, window_s = None, 30.0
    next_onset = 0
    for gid, cap in enumerate(captures):
        n = len(cap)
        if feature_names is None:
            feature_names, window_s = list(cap.feature_names), float(cap.window_size_s)
        elif list(cap.feature_names) != feature_names:
            raise ValueError("captures disagree on the feature space")
        a = (np.asarray(cap.infiltration).astype(np.int8) if late_idx is None
             else np.isin(np.asarray(cap.stage), late_idx).astype(np.int8))
        valid = np.asarray(cap.stage_mask).astype(bool)
        onset_ids: dict[int, int] = {}
        for t in range(max(context, quiet_gap) - 1, n - horizon):
            if a[t - quiet_gap + 1:t + 1].any():
                continue
            if not valid[t - quiet_gap + 1:t + horizon + 1].all():
                continue
            future = a[t + 1:t + 1 + horizon]
            hit = np.flatnonzero(future)
            ctx.append(cap.X[t - context + 1:t + 1])
            grp.append(gid + group_offset)
            tt.append(t)
            if len(hit):
                s = t + 1 + int(hit[0])
                if s not in onset_ids:
                    onset_ids[s] = next_onset
                    next_onset += 1
                y.append(1)
                steps.append(int(hit[0]) + 1)
                oid.append(onset_ids[s])
            else:
                y.append(0)
                steps.append(0)
                oid.append(-1)
    n_feat = len(feature_names or [])
    return OnsetSamples(
        context=(np.stack(ctx).astype(np.float32) if ctx
                 else np.zeros((0, context, n_feat), dtype=np.float32)),
        onset_in_k=np.asarray(y, dtype=np.int8), steps_to_onset=np.asarray(steps, dtype=np.int16),
        group=np.asarray(grp, dtype=np.int32), t=np.asarray(tt, dtype=np.int32),
        onset_id=np.asarray(oid, dtype=np.int64), window_size_s=window_s, horizon=horizon,
        quiet_gap=quiet_gap, feature_names=feature_names or [],
    )


def split_groups(samples: OnsetSamples, val_frac: float = 0.2, test_frac: float = 0.3,
                 seed: int = 0):
    """Split by capture -- never by sample -- so neighbouring windows cannot straddle a split."""
    groups = np.unique(samples.group)
    rng = np.random.default_rng(seed)
    order = rng.permutation(groups)
    n_test = max(1, int(round(len(order) * test_frac)))
    n_val = max(1, int(round(len(order) * val_frac)))
    test_g, val_g = order[:n_test], order[n_test:n_test + n_val]
    train_g = order[n_test + n_val:]
    pick = lambda gs: np.flatnonzero(np.isin(samples.group, gs))
    return samples.subset(pick(train_g)), samples.subset(pick(val_g)), samples.subset(pick(test_g))


# --------------------------------------------------------------------- scoring
def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on ``labels == 0`` is <= ``fpr``."""
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(neg)))          # how many negatives may sit at/above threshold
    if k >= len(neg):
        return float(neg[0])
    cut = neg[len(neg) - k - 1] if k < len(neg) else neg[0]
    # strictly above the (k+1)-th largest negative => at most k negatives alert
    return float(np.nextafter(cut, np.inf))


def _auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _ap(scores, labels):
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _event_stats(samples: OnsetSamples, alerts: np.ndarray):
    """Per-onset: was it flagged before it began, and how many seconds early."""
    ids = np.unique(samples.onset_id[samples.onset_id >= 0])
    if len(ids) == 0:
        return float("nan"), float("nan"), 0
    warned, leads = 0, []
    for i in ids:
        m = samples.onset_id == i
        hit = m & alerts
        if hit.any():
            warned += 1
            leads.append(float(samples.steps_to_onset[hit].max()) * samples.window_size_s)
    recall = warned / len(ids)
    return recall, (float(np.median(leads)) if leads else float("nan")), len(ids)


def _point_metrics(samples: OnsetSamples, scores: np.ndarray, threshold: float) -> dict:
    y = samples.onset_in_k.astype(bool)
    alerts = scores >= threshold
    neg, pos = ~y, y
    fpr = float(alerts[neg].mean()) if neg.any() else float("nan")
    tpr = float(alerts[pos].mean()) if pos.any() else float("nan")
    ev_recall, ev_lead, n_events = _event_stats(samples, alerts)
    out = {"threshold": float(threshold), "fpr": fpr, "tpr": tpr,
           "event_recall": ev_recall, "median_lead_s": ev_lead, "n_events": n_events,
           "false_alerts_per_day": (fpr * 86400.0 / samples.window_size_s
                                    if fpr == fpr else float("nan"))}
    for pi in PREVALENCES:
        denom = tpr * pi + fpr * (1 - pi)
        out[f"ppv@{pi:g}"] = float(tpr * pi / denom) if denom > 0 else float("nan")
    return out


def evaluate(samples: OnsetSamples, scores: np.ndarray, *, threshold: float | None = None,
             val_scores: np.ndarray | None = None, val_samples: OnsetSamples | None = None,
             fpr_budget: float = 1e-2, n_boot: int = 300, seed: int = 0) -> dict:
    """Score a forecaster on quiet samples.

    The operating threshold comes, in order of preference, from ``threshold``, from the
    ``fpr_budget`` quantile of ``val_scores`` (validation data, never test), and only
    as a last resort from the test negatives themselves (flagged ``threshold_source``).
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = samples.onset_in_k.astype(int)
    if threshold is not None:
        source = "given"
    elif val_scores is not None and val_samples is not None:
        threshold, source = threshold_for_fpr(np.asarray(val_scores, dtype=np.float64),
                                              val_samples.onset_in_k.astype(int), fpr_budget), "validation"
    else:
        threshold, source = threshold_for_fpr(scores, y, fpr_budget), "test (optimistic)"

    n_neg = int((y == 0).sum())
    res = {
        "n": len(samples), "n_pos": int(y.sum()), "n_neg": n_neg,
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "auc": _auc(scores, y), "ap": _ap(scores, y),
        "chance_ap": float(y.mean()) if len(y) else float("nan"),
        "operating": _point_metrics(samples, scores, threshold),
        "threshold_source": source,
        "fpr_budget": fpr_budget,
        # rule of three: with n negatives and no false alert, FPR could still be up to 3/n
        "fpr_resolution": 3.0 / n_neg if n_neg else float("nan"),
        "budget_resolvable": bool(n_neg and fpr_budget >= 3.0 / n_neg),
    }
    tpr_at = {}
    for b in FPR_BUDGETS:
        t = threshold_for_fpr(scores, y, b)
        tpr_at[f"{b:g}"] = float((scores[y == 1] >= t).mean()) if y.sum() else float("nan")
    res["tpr_at_fpr_test"] = tpr_at

    if n_boot and len(np.unique(samples.group)) >= 3:
        rng = np.random.default_rng(seed)
        groups = np.unique(samples.group)
        index_by_group = {g: np.flatnonzero(samples.group == g) for g in groups}
        aucs, recalls, tprs = [], [], []
        for _ in range(n_boot):
            pick = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([index_by_group[g] for g in pick])
            yy = y[idx]
            if yy.min() == yy.max():
                continue
            aucs.append(_auc(scores[idx], yy))
            al = scores[idx] >= threshold
            tprs.append(float(al[yy == 1].mean()))
            recalls.append(_event_stats(_dedupe_events(samples, idx, pick, index_by_group), al)[0])
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [float("nan")] * 2
        res["ci95"] = {"auc": ci(aucs), "tpr": ci(tprs), "event_recall": ci(recalls)}
    return res


def _dedupe_events(samples, idx, pick, index_by_group):
    """Bootstrap draws can repeat a capture; give each repeat its own onset ids."""
    sub = samples.subset(idx)
    oid = sub.onset_id.copy()
    cursor = 0
    for rep, g in enumerate(pick):
        n = len(index_by_group[g])
        seg = slice(cursor, cursor + n)
        oid[seg] = np.where(oid[seg] >= 0, oid[seg] + rep * 10_000_000, -1)
        cursor += n
    sub.onset_id = oid
    return sub


def contexts_from_states(states, context: int) -> np.ndarray:
    """Every length-``context`` sliding window of a capture, for false-alarm counting."""
    X = np.asarray(states.X, dtype=np.float32)
    if len(X) < context:
        return np.zeros((0, context, X.shape[1]), dtype=np.float32)
    return np.stack([X[t - context + 1:t + 1] for t in range(context - 1, len(X))])
