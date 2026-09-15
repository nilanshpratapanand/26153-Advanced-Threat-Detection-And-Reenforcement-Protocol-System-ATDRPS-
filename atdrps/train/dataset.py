"""Turn windowed captures into supervised sequences, and split them honestly.

Two things here decide whether every number the project reports is meaningful.

**The target is the future, not the present.**  Sample ``t`` is built from the
states ``S_{t-L+1..t}`` and is supervised on ``S_{t+1}`` and on the stage and
infiltration labels of windows ``t+1 .. t+K``.  A model trained to label the
window it was just shown is a classifier wearing a sequence model's clothes.

**Splits are by time, never at random.**  Adjacent windows overlap in content;
a random split puts a window's own neighbours in the training set and inflates
every metric.  Splits are taken chronologically, within each capture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..data.schema import STAGES
from ..data.windows import WindowedStates

__all__ = ["SequenceDataset", "build_sequences", "time_split", "group_split",
           "class_weights"]


@dataclass
class SequenceDataset:
    context: np.ndarray        # (N, L, F) observed states
    target_state: np.ndarray   # (N, F)    the next state -- the dynamics target
    target_stage: np.ndarray   # (N, K)    MITRE stage for windows t+1..t+K
    target_infil: np.ndarray   # (N, K)    infiltration flag for the same
    target_mask: np.ndarray    # (N, K)    False => that label must not train anything
    ts: np.ndarray             # (N,)      start time of window t
    group: np.ndarray          # (N,)      which capture the sample came from
    feature_names: list[str]
    stage_names: list[str]
    window_size_s: float = 30.0

    def __len__(self) -> int:
        return int(self.context.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.context.shape[2])

    @property
    def horizon(self) -> int:
        return int(self.target_stage.shape[1])

    @property
    def context_length(self) -> int:
        return int(self.context.shape[1])

    def subset(self, idx: np.ndarray) -> "SequenceDataset":
        idx = np.asarray(idx, dtype=np.int64)
        return replace(
            self,
            context=self.context[idx], target_state=self.target_state[idx],
            target_stage=self.target_stage[idx], target_infil=self.target_infil[idx],
            target_mask=self.target_mask[idx], ts=self.ts[idx], group=self.group[idx],
        )

    def describe(self) -> str:
        stages = self.target_stage[:, 0][self.target_mask[:, 0]]
        counts = {STAGES[i]: int((stages == i).sum()) for i in np.unique(stages)}
        return (f"{len(self)} sequences, context={self.context_length}, "
                f"horizon={self.horizon}, features={self.n_features}, "
                f"next-window stages {counts}")


def build_sequences(
    captures: list[WindowedStates],
    context: int = 16,
    horizon: int = 5,
) -> SequenceDataset:
    """Build (context -> future) samples from one or more windowed captures.

    Sequences never span two captures: a sample needs ``context`` real windows
    before it and ``horizon`` after it, all from the same capture.
    """
    if context < 1 or horizon < 1:
        raise ValueError("context and horizon must both be >= 1")

    ctx_list, state_list, stage_list, infil_list, mask_list = [], [], [], [], []
    ts_list, group_list = [], []
    feature_names: list[str] | None = None
    window_size = 30.0

    for gid, cap in enumerate(captures):
        if len(cap) < context + horizon:
            continue
        if feature_names is None:
            feature_names = list(cap.feature_names)
            window_size = cap.window_size_s
        elif list(cap.feature_names) != feature_names:
            raise ValueError("captures disagree on the feature space")

        X = cap.X
        n = len(cap)
        for t in range(context - 1, n - horizon):
            ctx_list.append(X[t - context + 1:t + 1])
            state_list.append(X[t + 1])
            future = slice(t + 1, t + 1 + horizon)
            stage_list.append(cap.stage[future])
            infil_list.append(cap.infiltration[future])
            mask_list.append(cap.stage_mask[future])
            ts_list.append(cap.ts_start[t])
            group_list.append(gid)

    if not ctx_list:
        n_features = len(captures[0].feature_names) if captures else 0
        names = list(captures[0].feature_names) if captures else []
        return SequenceDataset(
            context=np.zeros((0, context, n_features), dtype=np.float32),
            target_state=np.zeros((0, n_features), dtype=np.float32),
            target_stage=np.zeros((0, horizon), dtype=np.int64),
            target_infil=np.zeros((0, horizon), dtype=np.int64),
            target_mask=np.zeros((0, horizon), dtype=bool),
            ts=np.zeros(0), group=np.zeros(0, dtype=np.int64),
            feature_names=names, stage_names=list(STAGES),
            window_size_s=window_size,
        )

    return SequenceDataset(
        context=np.stack(ctx_list).astype(np.float32),
        target_state=np.stack(state_list).astype(np.float32),
        target_stage=np.stack(stage_list).astype(np.int64),
        target_infil=np.stack(infil_list).astype(np.int64),
        target_mask=np.stack(mask_list).astype(bool),
        ts=np.asarray(ts_list, dtype=np.float64),
        group=np.asarray(group_list, dtype=np.int64),
        feature_names=feature_names or [],
        stage_names=list(STAGES),
        window_size_s=window_size,
    )


def time_split(
    dataset: SequenceDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    gap: int = 0,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Chronological train / validation / test split, taken inside each capture.

    ``gap`` drops that many samples from the end of each block.  Consecutive
    samples share ``context - 1`` windows, so without a gap the last training
    sample and the first validation sample are almost the same observation.
    Setting ``gap >= context`` removes that overlap entirely.

    The gap is clamped so that it can never empty a block: on a short capture a
    generous gap would otherwise silently produce a validation set of size zero,
    and early stopping against an empty set is not a thing that fails loudly.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be fractions summing to < 1")

    train_idx, val_idx, test_idx = [], [], []
    for gid in np.unique(dataset.group):
        members = np.flatnonzero(dataset.group == gid)
        members = members[np.argsort(dataset.ts[members], kind="stable")]
        n = members.size
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        if n_train <= 0:
            continue
        # never let the gap consume a whole block
        g = max(0, min(gap, n_train - 1,
                       (n_val - 1) if n_val else 0,
                       (n_test - 1) if n_test else 0))
        train_end = n_train
        val_end = n_train + n_val
        train_idx.append(members[:train_end - g])
        if n_val:
            val_idx.append(members[train_end:val_end - g])
        if n_test:
            test_idx.append(members[val_end:])

    def stack(parts):
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    return (dataset.subset(stack(train_idx)),
            dataset.subset(stack(val_idx)),
            dataset.subset(stack(test_idx)))


def group_split(
    dataset: SequenceDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 1337,
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Hold out whole captures for validation and test.

    Captures are independent deployments, so this is the stricter split: train
    and test share no window, no flow and no campaign, which a chronological
    split inside one capture cannot claim (adjacent windows there overlap in
    content even with a gap).

    It also fixes a practical problem. Taking the final slice of each capture
    puts whichever stages happen to run late into test and everything else into
    train, so the positive class can almost vanish from the test set and the
    resulting F1 measures nothing. Whole captures carry whole campaigns.

    The assignment is seeded and therefore reproducible; it is not a random
    split of samples, which would leak neighbouring windows across the boundary.
    """
    groups = np.unique(dataset.group)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n = shuffled.size
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    if n_test + n_val >= n:
        raise ValueError(
            f"{n} captures cannot support {n_val} validation + {n_test} test groups"
        )
    test_groups = set(shuffled[:n_test].tolist())
    val_groups = set(shuffled[n_test:n_test + n_val].tolist())

    def pick(selector):
        return np.flatnonzero(np.array([selector(g) for g in dataset.group]))

    return (
        dataset.subset(pick(lambda g: g not in test_groups and g not in val_groups)),
        dataset.subset(pick(lambda g: g in val_groups)),
        dataset.subset(pick(lambda g: g in test_groups)),
    )


def class_weights(labels: np.ndarray, n_classes: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Inverse-frequency weights.

    Advanced stages are rare by nature -- exfiltration is a handful of windows
    in a capture full of benign traffic -- and unweighted training answers
    "benign" to everything and scores well doing it.
    """
    labels = np.asarray(labels).ravel()
    if mask is not None:
        labels = labels[np.asarray(mask).ravel().astype(bool)]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(n_classes, dtype=np.float64)
    if present.sum() == 0:
        return weights
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights
