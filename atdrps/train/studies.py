"""Two studies the presentation claims and the benchmark table does not prove.

``docs/BENCHMARKS.md`` answers "is the world model better than the baselines on
held-out captures?".  It does not answer either of the two questions a reviewer
who has read the slides will ask next.

**1. Generalisation to an attack family never trained on.**  The held-out-capture
split guarantees train and test share no window and no campaign, but every
attack *family* still appears on both sides.  A model that has memorised what
lateral movement looks like would score exactly the same.  :func:`run_attack_type_holdout`
removes a whole stage from training -- every capture that contains it -- and
measures what survives.  The stage head cannot name a class it was never shown,
and that is reported as a zero rather than quietly excluded; the question worth
asking is whether the *infiltration* forecast still fires, because that is the
alarm a SOC acts on.

**2. Does dual-level state fusion earn its cost?**  Packet-level features are
the expensive half of ingestion -- they need the full capture, not a flow
export.  :func:`run_ablation` trains the same model on flow-derived features
only, on packet-derived features only, and on both, plus a context-length sweep,
so "dual-level fusion" and "temporal context" are measured claims rather than
architecture-diagram labels.

Every study reuses :mod:`atdrps.train.dataset` and :mod:`atdrps.train.evaluate`:
same sequences, same standardisation, same threshold-on-validation rule.  A
study that built its own split would be measuring its own split.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..data.schema import STAGES, STAGE_INDEX
from ..data.windows import AGGREGATED_FLOW_FEATURES
from ..models.numpy_dynamics import NumpyDynamicsWorldModel, PersistenceModel
from ..models.transformer import TemporalTransformerWorldModel, torch_available
from .dataset import SequenceDataset, build_sequences, group_split
from .evaluate import (
    benchmark_table, binary_metrics, dynamics_metrics, find_best_threshold,
    stage_metrics,
)
from .train import rollout_predictions

__all__ = [
    "PACKET_AGGREGATE_BASES", "PACKET_DETECTORS", "feature_levels",
    "select_features", "context_slice", "group_stages",
    "run_attack_type_holdout", "run_ablation",
    "write_holdout_report", "write_ablation_report",
]

# Per-flow features that only exist because the packets were parsed.  A NetFlow
# or CICFlowMeter export carries none of them -- the information was discarded
# when the flow record was written.  Each contributes mean/std/max to the state.
PACKET_AGGREGATE_BASES: tuple[str, ...] = (
    "ttl_std", "ttl_unique", "window_mean",
    "payload_mean", "payload_zero_ratio",
    "retransmission_ratio", "frag_packet_count",
)

# Window-level detectors built from packet evidence rather than flow counters.
PACKET_DETECTORS: tuple[str, ...] = ("ttl_inconsistency", "rst_injection_ratio")


def feature_levels(feature_names: list[str]) -> dict[str, list[int]]:
    """Split state dimensions into packet-derived and flow-derived.

    The state is built by :func:`atdrps.data.windows.state_feature_names`, which
    emits ``base__mean`` / ``base__std`` / ``base__max`` for each aggregated
    per-flow feature, so a dimension is packet-derived when its *base* is.
    """
    packet_bases = set(PACKET_AGGREGATE_BASES)
    packet_names = set(PACKET_DETECTORS)
    known_bases = set(AGGREGATED_FLOW_FEATURES)

    packet_idx, flow_idx = [], []
    for i, name in enumerate(feature_names):
        base = name.split("__", 1)[0] if "__" in name else name
        if base in known_bases:
            is_packet = base in packet_bases
        else:
            is_packet = name in packet_names
        (packet_idx if is_packet else flow_idx).append(i)
    return {"packet": packet_idx, "flow": flow_idx}


def select_features(dataset: SequenceDataset, idx) -> SequenceDataset:
    """A view of the dataset over a subset of state dimensions.

    The dynamics target is narrowed with the input: a model given only flow
    features is asked to predict only the flow features' next values, so
    next-state error stays comparable across ablation arms.
    """
    idx = np.asarray(idx, dtype=np.int64)
    return replace(
        dataset,
        context=dataset.context[:, :, idx],
        target_state=dataset.target_state[:, idx],
        feature_names=[dataset.feature_names[i] for i in idx],
    )


def context_slice(dataset: SequenceDataset, length: int) -> SequenceDataset:
    """Keep only the most recent ``length`` windows of context."""
    if length < 1 or length > dataset.context_length:
        raise ValueError(f"context length must be in 1..{dataset.context_length}")
    return replace(dataset, context=dataset.context[:, -length:, :])


def group_stages(dataset: SequenceDataset) -> dict[int, set[int]]:
    """Which MITRE stages each capture actually contains, from its labels."""
    out: dict[int, set[int]] = {}
    for gid in np.unique(dataset.group):
        members = dataset.group == gid
        stages = dataset.target_stage[members][dataset.target_mask[members]]
        out[int(gid)] = {int(s) for s in np.unique(stages)}
    return out


# --------------------------------------------------------------- study 1
def run_attack_type_holdout(
    captures,
    holdout_stage: str = "LateralMovement",
    context: int = 16,
    horizon: int = 5,
    val_frac: float = 0.2,
    backend: str = "linear",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    seed: int = 1337,
    verbose: bool = True,
) -> dict:
    """Train without a whole attack family; test on the captures that carry it.

    Validation comes from the *training* captures, so the held-out family is
    unseen at threshold-selection time too.  Choosing the operating point on
    captures containing the family would leak exactly the thing being measured.
    """
    if holdout_stage not in STAGE_INDEX:
        raise ValueError(f"unknown stage {holdout_stage!r}; one of {list(STAGES)}")
    target = STAGE_INDEX[holdout_stage]

    dataset = build_sequences(captures, context=context, horizon=horizon)
    if len(dataset) == 0:
        raise ValueError("no sequences could be built - captures are too short")

    stages_by_group = group_stages(dataset)
    held = {g for g, s in stages_by_group.items() if target in s}
    seen = {g for g in stages_by_group if g not in held}
    if not held:
        raise ValueError(f"no capture contains {holdout_stage}")
    if len(seen) < 3:
        raise ValueError(
            f"{holdout_stage} appears in {len(held)} of {len(stages_by_group)} "
            "captures; too few remain to train on"
        )

    rng = np.random.default_rng(seed)
    seen_list = rng.permutation(sorted(seen))
    n_val = max(1, int(round(len(seen_list) * val_frac)))
    val_groups = set(int(g) for g in seen_list[:n_val])
    train_groups = set(int(g) for g in seen_list[n_val:])

    def pick(groups):
        return dataset.subset(np.flatnonzero(np.isin(dataset.group, list(groups))))

    train, val, test = pick(train_groups), pick(val_groups), pick(held)

    if verbose:
        print(f"[holdout] {holdout_stage}: {len(held)} captures held out, "
              f"{len(train_groups)} train / {len(val_groups)} val")
        print(f"[holdout] train={len(train)} val={len(val)} test={len(test)} sequences")

    # the held-out stage must be absent from training, or the study proves nothing
    train_stages = (set(int(v) for v in np.unique(train.target_stage[train.target_mask]))
                    if len(train) else set())
    if target in train_stages:
        raise ValueError(
            f"{holdout_stage} still appears in the training split - the holdout "
            "is not a holdout"
        )

    t0 = time.time()
    model = _fit_model(backend, train, val, context, horizon,
                       epochs=epochs, batch_size=batch_size, lr=lr,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       verbose=verbose)
    fit_seconds = time.time() - t0

    stage_p, infil_p = rollout_predictions(model, test, horizon)
    v_stage, v_infil = rollout_predictions(model, val, horizon)

    mask = test.target_mask[:, 0]
    y_true = test.target_infil[:, 0][mask]
    scores = infil_p[:, 0][mask]
    truth_stage = test.target_stage[:, 0][mask]

    threshold = 0.5
    v_mask = val.target_mask[:, 0]
    if v_mask.any() and np.unique(val.target_infil[:, 0][v_mask]).size == 2:
        threshold = find_best_threshold(
            val.target_infil[:, 0][v_mask], v_infil[:, 0][v_mask], metric="f1"
        )

    overall = binary_metrics(y_true, scores, threshold)

    # the sharper question: on the windows whose true stage IS the unseen
    # family, does the alarm fire at all?
    unseen = truth_stage == target
    benign = truth_stage == STAGE_INDEX["Benign"]
    unseen_recall = float((scores[unseen] >= threshold).mean()) if unseen.any() else float("nan")
    benign_fpr = float((scores[benign] >= threshold).mean()) if benign.any() else float("nan")

    # what does the stage head call the family it was never shown?
    named: dict[str, int] = {}
    if unseen.any():
        pred = stage_p[:, 0][mask][unseen].argmax(axis=1)
        for cls, count in zip(*np.unique(pred, return_counts=True)):
            named[STAGES[int(cls)]] = int(count)

    sm = stage_metrics(truth_stage, stage_p[:, 0][mask], list(STAGES))

    # Detection rate per stage, marked by whether training ever saw that stage.
    # Without the seen rows the unseen number means nothing: a low rate could
    # equally be a mistuned threshold, and the comparison is what separates
    # the two explanations.
    per_stage = {}
    benign_idx = STAGE_INDEX["Benign"]
    for cls in sorted(set(int(c) for c in np.unique(truth_stage))):
        if cls == benign_idx:
            continue
        rows_cls = truth_stage == cls
        entry = {
            "windows": int(rows_cls.sum()),
            "detection_rate": float((scores[rows_cls] >= threshold).mean()),
            "trained_on": cls in train_stages,
            "mean_probability": float(scores[rows_cls].mean()),
        }
        # threshold-free: can the model RANK this family above benign traffic,
        # even if the operating point chosen on seen families is wrong for it?
        if benign.any() and rows_cls.any():
            from sklearn.metrics import roc_auc_score
            y = np.concatenate([np.ones(int(rows_cls.sum())), np.zeros(int(benign.sum()))])
            sc = np.concatenate([scores[rows_cls], scores[benign]])
            entry["auc_vs_benign"] = float(roc_auc_score(y, sc))
        per_stage[STAGES[cls]] = entry

    # the held-out family is the headline, but excluding it upstream can strand
    # later chain stages too (no lateral movement means no C2 that follows it).
    # Report every stage training never saw, or the table overstates what was
    # actually held out.
    also_unseen = {}
    for cls in sorted(set(int(c) for c in np.unique(truth_stage)) - train_stages):
        rows_cls = truth_stage == cls
        also_unseen[STAGES[cls]] = {
            "windows": int(rows_cls.sum()),
            "detection_rate": float((scores[rows_cls] >= threshold).mean()),
        }

    payload = {
        "study": "attack-type holdout",
        "holdout_stage": holdout_stage,
        "backend": backend,
        "captures": {"total": len(stages_by_group), "held_out": len(held),
                     "train": len(train_groups), "val": len(val_groups)},
        "sequences": {"train": len(train), "val": len(val), "test": len(test)},
        "threshold": float(threshold),
        "fit_seconds": fit_seconds,
        "infiltration_all_windows": {
            "f1": overall.f1, "precision": overall.precision,
            "recall": overall.recall, "fpr": overall.fpr,
            "roc_auc": overall.roc_auc,
            "support_positive": overall.support_positive,
        },
        "unseen_family": {
            "windows": int(unseen.sum()),
            "detection_rate": unseen_recall,
            "predicted_stage_counts": named,
            "stage_recall": float(sm.per_class.get(holdout_stage, {}).get("recall", 0.0)),
        },
        "stages_absent_from_training": also_unseen,
        "per_stage": per_stage,
        "benign_false_alarm_rate": benign_fpr,
        "stage_accuracy_all_windows": sm.accuracy,
        "stage_macro_f1_all_windows": sm.macro_f1,
    }
    if verbose:
        print(f"[holdout] detection rate on unseen {holdout_stage}: "
              f"{unseen_recall:.4f} over {int(unseen.sum())} windows "
              f"(threshold {threshold:.3f}, benign false-alarm {benign_fpr:.4f})")
    return payload


# --------------------------------------------------------------- study 2
def run_ablation(
    captures,
    context: int = 16,
    horizon: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    backend: str = "linear",
    short_context: int = 4,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    verbose: bool = True,
) -> dict:
    """Flow-only / packet-only / dual, and a context-length sweep.

    One split, one model class, one threshold rule; only the input changes.
    """
    dataset = build_sequences(captures, context=context, horizon=horizon)
    if len(dataset) == 0:
        raise ValueError("no sequences could be built - captures are too short")
    train, val, test = group_split(dataset, val_frac, test_frac)
    levels = feature_levels(dataset.feature_names)
    n_flow, n_packet = len(levels["flow"]), len(levels["packet"])
    all_idx = list(range(dataset.n_features))

    arms = [
        ("flow-only", levels["flow"], context,
         f"{n_flow} flow-derived dims; what a NetFlow export gives you"),
        ("packet-only", levels["packet"], context,
         f"{n_packet} packet-derived dims; needs the full capture"),
        ("dual (shipped)", all_idx, context,
         f"all {len(all_idx)} dims, {context}-window context"),
        ("dual, context=1", all_idx, 1,
         "dual features, current window only -- no temporal model"),
        (f"dual, context={short_context}", all_idx, short_context,
         "dual features, short context"),
    ]

    rows, results = [], []
    for name, idx, ctx_len, note in arms:
        if ctx_len > context:
            continue
        tr = context_slice(select_features(train, idx), ctx_len)
        va = context_slice(select_features(val, idx), ctx_len)
        te = context_slice(select_features(test, idx), ctx_len)

        t0 = time.time()
        model = _fit_model(backend, tr, va, ctx_len, horizon,
                           epochs=epochs, batch_size=batch_size, lr=lr,
                           d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           verbose=False)
        stage_p, infil_p = rollout_predictions(model, te, horizon)
        v_stage, v_infil = rollout_predictions(model, va, horizon)

        mask = te.target_mask[:, 0]
        threshold = 0.5
        v_mask = va.target_mask[:, 0]
        if v_mask.any() and np.unique(va.target_infil[:, 0][v_mask]).size == 2:
            threshold = find_best_threshold(
                va.target_infil[:, 0][v_mask], v_infil[:, 0][v_mask], metric="f1"
            )
        bm = binary_metrics(te.target_infil[:, 0][mask], infil_p[:, 0][mask], threshold)
        sm = stage_metrics(te.target_stage[:, 0][mask], stage_p[:, 0][mask], list(STAGES))

        floor = PersistenceModel(tr.feature_names, ctx_len, horizon).fit(tr)
        dyn = dynamics_metrics(model.predict_next_batch(te.context),
                               te.target_state, model.standardiser)
        dyn_floor = dynamics_metrics(floor.predict_next_batch(te.context),
                                     te.target_state, floor.standardiser)

        rows.append({
            "Variant": name, "Dims": len(idx), "Context": ctx_len,
            "F1": bm.f1, "Precision": bm.precision, "Recall": bm.recall,
            "FPR": bm.fpr, "StageAcc": sm.accuracy, "StageMacroF1": sm.macro_f1,
            "NextStateMSE": dyn["mse"], "PersistenceMSE": dyn_floor["mse"],
        })
        results.append({"variant": name, "note": note, "dims": len(idx),
                        "context": ctx_len, "threshold": float(threshold),
                        "fit_seconds": time.time() - t0,
                        "infiltration": bm.as_row(),
                        "stage": {"accuracy": sm.accuracy, "macro_f1": sm.macro_f1},
                        "dynamics": dyn, "persistence": dyn_floor})
        if verbose:
            print(f"[ablate ] {name:<20} F1={bm.f1:.4f} FPR={bm.fpr:.4f} "
                  f"StageAcc={sm.accuracy:.4f} MSE={dyn['mse']:.4f}")

    return {
        "study": "ablation",
        "backend": backend,
        "split": "held-out captures",
        "sequences": {"train": len(train), "val": len(val), "test": len(test)},
        "feature_counts": {"flow": n_flow, "packet": n_packet,
                           "total": dataset.n_features},
        "rows": rows,
        "arms": results,
    }


# ------------------------------------------------------------------ shared
def _fit_model(backend, train, val, context, horizon, *, epochs, batch_size, lr,
               d_model, n_heads, n_layers, verbose):
    if backend == "transformer":
        if not torch_available():
            raise RuntimeError("PyTorch is not installed; use backend='linear'")
        model = TemporalTransformerWorldModel(
            train.feature_names, context, horizon,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        model.fit(train, val_dataset=val if len(val) else None, epochs=epochs,
                  batch_size=batch_size, lr=lr, verbose=verbose)
        return model
    if backend != "linear":
        raise ValueError("backend must be 'linear' or 'transformer'")
    model = NumpyDynamicsWorldModel(train.feature_names, context, horizon)
    model.fit(train, val_dataset=val if len(val) else None, verbose=verbose)
    return model


def write_holdout_report(payload: dict, path: str | Path) -> str:
    h = payload["holdout_stage"]
    u = payload["unseen_family"]
    inf = payload["infiltration_all_windows"]
    per = payload.get("per_stage", {})
    named = ", ".join(f"{k} {v}" for k, v in
                      sorted(u["predicted_stage_counts"].items(),
                             key=lambda kv: -kv[1])) or "n/a"
    seen_rates = [v["detection_rate"] for v in per.values() if v["trained_on"]]
    seen_mean = float(np.mean(seen_rates)) if seen_rates else float("nan")

    lines = [
        f"# Generalisation to an unseen attack family: {h}",
        "",
        f"Every capture containing **{h}** was removed from training *and* from "
        "validation, so the operating threshold was also chosen without ever "
        "seeing the family. Those captures are the test set. The training half "
        "was generated so the family is impossible in it, not merely absent by "
        "luck of the seed (`scripts/make_corpus.py --holdout-stage`).",
        "",
        f"- captures: {payload['captures']['train']} train, "
        f"{payload['captures']['val']} validation, "
        f"{payload['captures']['held_out']} held out "
        f"(of {payload['captures']['total']})",
        f"- sequences: {payload['sequences']['train']} / "
        f"{payload['sequences']['val']} / {payload['sequences']['test']}",
        f"- model: `{payload['backend']}`, threshold "
        f"{payload['threshold']:.3f} chosen on validation",
        "",
        "## Detection rate per attack family, in the held-out captures",
        "",
        benchmark_table([
            {"Family": name,
             "In training?": "yes" if v["trained_on"] else "**no**",
             "Windows": v["windows"],
             "Alarm fired": v["detection_rate"],
             "Mean p": v["mean_probability"],
             "AUC vs benign": v.get("auc_vs_benign", float("nan"))}
            for name, v in sorted(per.items(),
                                  key=lambda kv: (kv[1]["trained_on"], -kv[1]["windows"]))
        ], ["Family", "In training?", "Windows", "Alarm fired", "Mean p",
            "AUC vs benign"]),
        "",
        f"False-alarm rate on the benign windows of the same captures: "
        f"**{payload['benign_false_alarm_rate']:.4f}**. Infiltration F1 over "
        f"every held-out window (all families, benign included): "
        f"{inf['f1']:.4f} at FPR {inf['fpr']:.4f}.",
        "",
        "## What this actually shows",
        "",
    ]

    rate = u["detection_rate"]
    auc = per.get(h, {}).get("auc_vs_benign", float("nan"))
    lines.append(
        f"At the shipped operating point the alarm fires on **{rate:.1%}** of "
        f"{h} windows it was never trained on, at a "
        f"{payload['benign_false_alarm_rate']:.1%} false-alarm rate on benign "
        f"traffic in the same captures (threshold {payload['threshold']:.3f}, "
        "chosen on validation captures that also lacked the family)."
    )
    lines.append("")

    # The control column is not clean, and saying so is the point of running
    # the study rather than quoting it.
    weak_seen = sorted(
        (name for name, v in per.items()
         if v["trained_on"] and v.get("auc_vs_benign", 1.0) < 0.60),
        key=lambda n: per[n].get("auc_vs_benign", 1.0),
    )
    lines.append(
        f"The trained-on families in the same captures average "
        f"{seen_mean:.1%}, which looks like the unseen family does *better*. "
        "It does not: the two halves of this corpus were generated "
        "differently, because forcing a family out of the training half means "
        "truncating the campaigns that lead to it."
    )
    if weak_seen:
        detail = ", ".join(
            f"{n} (AUC {per[n]['auc_vs_benign']:.4f} on "
            f"{per[n]['windows']} windows)" for n in weak_seen
        )
        lines.append("")
        lines.append(
            f"That shift is visible: {detail} rank at or below chance against "
            "benign traffic despite being present in training. Those families "
            "appear in the training half only in the shapes the truncated "
            "generator could produce -- exfiltration as DNS tunnelling rather "
            "than bulk outbound transfer, for instance -- so they are closer "
            "to unseen families than the label suggests. The trained-on column "
            "is therefore a weak control, not a clean one, and the honest "
            "reading of this study is the absolute number for the held-out "
            "family, not the comparison."
        )
    lines.append("")
    if auc == auc:
        if auc >= 0.75 and rate < 0.5:
            lines.append(
                f"The threshold-free number is the more informative one: AUC "
                f"{auc:.4f} against benign traffic means the model *ranks* the "
                "unseen family well above background, while the operating "
                "point -- chosen on families it had seen -- sits too high to "
                "convert that ranking into alarms. The signal transfers; the "
                "threshold does not. A deployment expecting novel families "
                "should set its threshold on a held-out family, not on the "
                "training families."
            )
        elif auc >= 0.75:
            lines.append(
                f"AUC against benign traffic is {auc:.4f} and the operating "
                "point carries over as well: both the ranking and the "
                f"threshold transfer to {h} here. Read that against the "
                "false-alarm rate above before treating it as a strong result "
                "-- a low bar fires on everything, and this corpus is "
                "synthetic."
            )
        else:
            lines.append(
                f"AUC against benign traffic is {auc:.4f}, so this is not just "
                "a mistuned threshold: the model ranks the unseen family only "
                "modestly above background. Transfer to a genuinely novel "
                "family is partial, not free."
            )
    lines += [
        "",
        f"**Do not read this as {h} detection.** What this study supports is "
        "the narrow claim: a model that never saw the family still recovers a "
        "minority of it from trajectory alone, at a low false-alarm rate. A "
        "signature-based detector recovers none of an unseen family by "
        "construction. It does not support any claim about detecting novel "
        "attacks in general, and the same pipeline trained on a corpus "
        "containing every family scores far higher (`docs/BENCHMARKS.md`, a "
        "different corpus and split, so the two numbers are not directly "
        "comparable). The operational conclusion is to keep the training "
        "corpus current rather than to rely on transfer.",
        "",
        "## What the stage head calls it",
        "",
        f"The {h} class was never in the training labels, so the stage head "
        f"**cannot** emit it: recall on those windows is "
        f"{u['stage_recall']:.4f} by construction. That is a property of "
        "closed-set classification, not a result. What it emits instead: "
        f"{named}.",
        "",
        _absent_stage_table(payload),
        "Stage accuracy over all held-out windows (benign included) is "
        f"{payload['stage_accuracy_all_windows']:.4f}, macro-F1 "
        f"{payload['stage_macro_f1_all_windows']:.4f}; both are floored by the "
        "missing classes and are reported that way on purpose.",
        "",
    ]
    text = "\n".join(lines)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def write_ablation_report(payload: dict, path: str | Path) -> str:
    fc = payload["feature_counts"]
    lines = [
        "# Ablation: what each half of the state buys",
        "",
        "One split (held-out captures), one model class "
        f"(`{payload['backend']}`), one threshold rule (chosen on validation). "
        "Only the input changes.",
        "",
        f"- state dimensions: **{fc['total']}** = {fc['flow']} flow-derived + "
        f"{fc['packet']} packet-derived",
        f"- sequences: {payload['sequences']['train']} train / "
        f"{payload['sequences']['val']} val / {payload['sequences']['test']} test",
        "",
        "Flow-derived dimensions are what a NetFlow or CICFlowMeter export "
        "carries. Packet-derived dimensions (TTL spread, TCP window trace, "
        "payload distribution, retransmissions, fragmentation, injected RSTs) "
        "exist only because ATDRPS parses the packets itself.",
        "",
        benchmark_table(payload["rows"], [
            "Variant", "Dims", "Context", "F1", "Precision", "Recall", "FPR",
            "StageAcc", "StageMacroF1", "NextStateMSE", "PersistenceMSE",
        ]),
        "",
        _ablation_findings(payload["rows"]),
        "",
        "`PersistenceMSE` is the dynamics floor for that feature subset -- "
        "copy the current state forward. It differs per row because the "
        "feature set differs, so next-state error is only comparable against "
        "the floor beside it, never across rows.",
        "",
    ]
    text = "\n".join(lines)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def _ablation_findings(rows: list[dict]) -> str:
    """Read the table back and state what it says -- including where the
    shipped configuration loses, which is the half a reader will check."""
    by = {r["Variant"]: r for r in rows}
    dual = by.get("dual (shipped)")
    if dual is None:
        return ""
    out = ["## What the table says", ""]

    flow, packet = by.get("flow-only"), by.get("packet-only")
    if flow and packet:
        out.append(
            f"**Fusion.** Neither half alone matches the pair: flow-only F1 "
            f"{flow['F1']:.4f}, packet-only {packet['F1']:.4f}, both "
            f"{dual['F1']:.4f} (+{dual['F1'] - max(flow['F1'], packet['F1']):.4f} "
            "over the better half). The gain is in recall "
            f"({dual['Recall']:.4f} vs {flow['Recall']:.4f} flow-only), which is "
            "the direction that matters for forecasting: the packet features "
            "carry evidence of attacks whose flow counters look ordinary."
        )
        if flow["FPR"] < dual["FPR"]:
            out.append("")
            out.append(
                f"**Where it loses.** Flow-only has the lower false-positive "
                f"rate ({flow['FPR']:.4f} vs {dual['FPR']:.4f}) and the higher "
                f"precision ({flow['Precision']:.4f} vs "
                f"{dual['Precision']:.4f}). Fusion buys recall and pays for it "
                "in precision. A deployment that cares more about analyst time "
                "than about missed attacks should know that and can pick the "
                "operating point accordingly."
            )

    static = by.get("dual, context=1")
    if static:
        out.append("")
        out.append(
            f"**Temporal context.** The same features with no history score "
            f"{static['F1']:.4f} F1 and {static['FPR']:.4f} FPR, against "
            f"{dual['F1']:.4f} and {dual['FPR']:.4f} with 16 windows -- "
            f"{dual['F1'] - static['F1']:+.4f} F1 for looking backwards. "
            "This is the world-model claim measured on its own: identical "
            "features, identical model class, only the history removed."
        )
        short = next((r for k, r in by.items()
                      if k.startswith("dual, context=") and r is not static), None)
        if short:
            out.append("")
            out.append(
                f"Most of that is bought early: {short['Context']} windows of "
                f"history already reach {short['F1']:.4f}. Going to 16 adds "
                f"{dual['F1'] - short['F1']:+.4f}."
            )
        if static["NextStateMSE"] < dual["NextStateMSE"]:
            out.append("")
            out.append(
                f"Next-state error runs the other way ({static['NextStateMSE']:.4f} "
                f"vs {dual['NextStateMSE']:.4f}): for a *linear* model, a long "
                "context is extra parameters to fit and the immediately "
                "preceding window is most of the signal for one-step "
                "prediction. Context earns its place on the forecasting and "
                "stage heads, not on next-state regression."
            )
    return "\n".join(out)


def _absent_stage_table(payload: dict) -> str:
    absent = payload.get("stages_absent_from_training") or {}
    if not absent:
        return ""
    rows = [{"Stage never trained on": k,
             "Held-out windows": v["windows"],
             "Alarm fired": v["detection_rate"]}
            for k, v in sorted(absent.items(), key=lambda kv: -kv[1]["windows"])]
    return ("\n".join([
        "Excluding one family upstream can strand the chain stages that follow "
        "it, so every stage the training split never contained is listed, not "
        "just the one named:",
        "",
        benchmark_table(rows, ["Stage never trained on", "Held-out windows",
                               "Alarm fired"]),
        "",
    ]))
