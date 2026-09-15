"""Train every model on identical data and produce the benchmark.

One entry point, :func:`run_benchmark`, because the comparison is only worth
anything if nothing differs between the models except the model.  Same
sequences, same chronological split, same standardisation, same thresholds
chosen on the same validation set, same rollout code.

Contestants:

* ``persistence``            -- "nothing changes".  The floor for dynamics.
* ``logreg-static``          -- logistic regression on the current window only.
* ``logreg-context``         -- logistic regression on the full L-window context.
* ``linear-dynamics``        -- ridge dynamics + logistic heads.
* ``temporal-transformer``   -- the world model, when PyTorch is available.

Operating thresholds come from the validation split.  Tuning a threshold on the
data you then report turns a benchmark into a sales pitch.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data.schema import STAGES
from ..models.baseline import LogisticBaseline
from ..models.numpy_dynamics import NumpyDynamicsWorldModel, PersistenceModel
from ..models.transformer import TemporalTransformerWorldModel, torch_available
from .dataset import build_sequences, group_split, time_split
from .evaluate import (
    benchmark_table, binary_metrics, dynamics_metrics, find_best_threshold,
    stage_metrics,
)

__all__ = ["run_benchmark", "rollout_predictions", "write_report"]


def rollout_predictions(model, dataset, horizon: int, batch: int = 256):
    """Run the model's own K-step forward simulation over a whole split.

    Deliberately uses :meth:`WorldModel.rollout` rather than a direct
    multi-horizon head, because that is what ships: the number reported is the
    number the deployed engine produces, compounding errors and all.
    """
    n = len(dataset)
    stage = np.zeros((n, horizon, len(STAGES)), dtype=np.float32)
    infil = np.zeros((n, horizon), dtype=np.float32)
    for i in range(n):
        forecast = model.rollout(dataset.context[i], horizon=horizon)
        stage[i] = forecast.stage_probs
        infil[i] = forecast.infiltration
    return stage, infil


def _evaluate(name, stage_probs, infil_scores, dataset, val_pack, horizon, extra=None):
    """Metrics for one model over every horizon step."""
    rows, per_step = [], []
    for k in range(horizon):
        mask = dataset.target_mask[:, k]
        if not mask.any():
            continue
        y_infil = dataset.target_infil[:, k][mask]
        scores = infil_scores[:, k][mask]

        threshold = 0.5
        if val_pack is not None:
            v_stage, v_infil, v_ds = val_pack
            v_mask = v_ds.target_mask[:, k]
            if v_mask.any() and np.unique(v_ds.target_infil[:, k][v_mask]).size == 2:
                threshold = find_best_threshold(
                    v_ds.target_infil[:, k][v_mask], v_infil[:, k][v_mask], metric="f1"
                )

        bm = binary_metrics(y_infil, scores, threshold)
        sm = stage_metrics(dataset.target_stage[:, k][mask], stage_probs[:, k][mask], list(STAGES))
        row = {"Model": name, "Step": k + 1}
        row.update(bm.as_row())
        row["StageAcc"] = sm.accuracy
        row["StageMacroF1"] = sm.macro_f1
        rows.append(row)
        per_step.append({
            "step": k + 1, "threshold": threshold,
            "infiltration": asdict(bm),
            "stage": {"accuracy": sm.accuracy, "macro_f1": sm.macro_f1,
                      "weighted_f1": sm.weighted_f1, "per_class": sm.per_class,
                      "labels": sm.labels, "confusion": sm.confusion.tolist()},
        })
    result = {"name": name, "steps": per_step}
    if extra:
        result.update(extra)
    return rows, result


def run_benchmark(
    captures,
    context: int = 16,
    horizon: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    gap: int | None = None,
    split: str = "group",
    out_dir: str | Path = "artifacts",
    use_torch: bool = True,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = context if gap is None else int(gap)

    dataset = build_sequences(captures, context=context, horizon=horizon)
    if len(dataset) == 0:
        raise ValueError("no sequences could be built - captures are too short")
    if split == "group":
        train, val, test = group_split(dataset, val_frac, test_frac)
        split_note = "held-out captures (train and test share no window or campaign)"
    elif split == "time":
        train, val, test = time_split(dataset, val_frac, test_frac, gap=gap)
        split_note = f"chronological within capture, {gap}-sample gap at each boundary"
    else:
        raise ValueError("split must be 'group' or 'time'")
    if verbose:
        print(f"[data] {dataset.describe()}")
        print(f"[data] train={len(train)} val={len(val)} test={len(test)}  [{split_note}]")
        for name, part in (("train", train), ("val", val), ("test", test)):
            if len(part):
                pos = int(part.target_infil[:, 0][part.target_mask[:, 0]].sum())
                print(f"[data]   {name}: {pos} infiltration-positive windows "
                      f"of {int(part.target_mask[:, 0].sum())} labelled")

    rows: list[dict] = []
    results: list[dict] = []
    models: dict = {}

    # ---------------------------------------------------- dynamics floor
    persistence = PersistenceModel(dataset.feature_names, context, horizon).fit(train)
    dyn = {
        "persistence": dynamics_metrics(
            persistence.predict_next_batch(test.context), test.target_state,
            persistence.standardiser,
        )
    }

    # --------------------------------------------------------- baselines
    for mode in ("static", "context"):
        t0 = time.time()
        base = LogisticBaseline(mode=mode).fit(train, n_stages=len(STAGES))
        stage_p, infil_p = base.predict(test)
        val_pack = (*base.predict(val), val) if len(val) else None
        r, res = _evaluate(f"logreg-{mode}", stage_p, infil_p, test, val_pack, horizon,
                           extra={"fit_seconds": time.time() - t0,
                                  "note": "direct multi-horizon; no forward simulation"})
        rows += r
        results.append(res)
        models[f"logreg-{mode}"] = base
        if verbose:
            print(f"[fit ] logreg-{mode} in {time.time() - t0:.1f}s")

    # --------------------------------------------- linear dynamics model
    t0 = time.time()
    linear = NumpyDynamicsWorldModel(dataset.feature_names, context, horizon)
    linear.fit(train, val_dataset=val if len(val) else None, verbose=verbose)
    dyn["linear-dynamics"] = dynamics_metrics(
        linear.predict_next_batch(test.context), test.target_state, linear.standardiser
    )
    stage_p, infil_p = rollout_predictions(linear, test, horizon)
    val_pack = (*rollout_predictions(linear, val, horizon), val) if len(val) else None
    r, res = _evaluate("linear-dynamics", stage_p, infil_p, test, val_pack, horizon,
                       extra={"fit_seconds": time.time() - t0,
                              "ridge_alpha": linear.ridge_alpha,
                              "note": "K-step autoregressive rollout"})
    rows += r
    results.append(res)
    models["linear-dynamics"] = linear
    if verbose:
        print(f"[fit ] linear-dynamics in {time.time() - t0:.1f}s "
              f"(alpha={linear.ridge_alpha:g})")

    # ------------------------------------------------ temporal transformer
    if use_torch and torch_available():
        t0 = time.time()
        world = TemporalTransformerWorldModel(
            dataset.feature_names, context, horizon,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        world.fit(train, val_dataset=val if len(val) else None, epochs=epochs,
                  batch_size=batch_size, lr=lr, verbose=verbose)
        dyn["temporal-transformer"] = dynamics_metrics(
            world.predict_next_batch(test.context), test.target_state, world.standardiser
        )
        stage_p, infil_p = rollout_predictions(world, test, horizon)
        val_pack = (*rollout_predictions(world, val, horizon), val) if len(val) else None
        r, res = _evaluate("temporal-transformer", stage_p, infil_p, test, val_pack, horizon,
                           extra={"fit_seconds": time.time() - t0,
                                  "epochs_run": len(world.history),
                                  "note": "K-step autoregressive rollout"})
        rows += r
        results.append(res)
        models["temporal-transformer"] = world
        world.save(out_dir / "model-transformer")
        if verbose:
            print(f"[fit ] temporal-transformer in {time.time() - t0:.1f}s")
    elif verbose:
        print("[skip] temporal-transformer: PyTorch is not installed in this "
              "environment (see requirements.txt)")

    linear.save(out_dir / "model-linear")

    payload = {
        "dataset": {
            "sequences": len(dataset), "train": len(train), "val": len(val),
            "test": len(test), "context": context, "horizon": horizon,
            "features": dataset.n_features, "window_size_s": dataset.window_size_s,
            "split": split_note, "split_kind": split, "gap": gap,
        },
        "dynamics": dyn,
        "models": results,
        "rows": rows,
        "torch_available": torch_available(),
    }
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"payload": payload, "models": models, "rows": rows,
            "splits": (train, val, test), "dataset": dataset}


def write_report(payload: dict, path: str | Path) -> str:
    """Render docs/BENCHMARKS.md from a benchmark payload."""
    rows = payload["rows"]
    meta = payload["dataset"]
    step1 = [r for r in rows if r["Step"] == 1]
    columns = ["Model", "Step", "F1", "Precision", "Recall", "FPR", "ROC-AUC",
               "StageAcc", "StageMacroF1"]

    dyn_rows = [
        {"Model": name, "Next-state MSE": m["mse"], "MAE": m["mae"], "RMSE": m["rmse"]}
        for name, m in payload["dynamics"].items()
    ]

    lines = [
        "# ATDRPS benchmark results",
        "",
        "All models see identical sequences, an identical split and identical "
        "features. Operating thresholds are chosen on the validation split and "
        "applied unchanged to test -- tuning a threshold on the data you then "
        "report turns a benchmark into a sales pitch.",
        "",
        "## Setup",
        "",
        f"- sequences: **{meta['sequences']}** "
        f"(train {meta['train']} / val {meta['val']} / test {meta['test']})",
        f"- context: **{meta['context']} windows** of {meta['window_size_s']:g}s; "
        f"forecast horizon **K = {meta['horizon']}**",
        f"- state dimension: **{meta['features']}**",
        f"- split: {meta['split']}"
        + (f", with a {meta['gap']}-sample gap at each boundary so that train and "
           "test never share a window" if meta.get("split_kind") == "time" else ""),
        "",
        "## Infiltration forecast, one window ahead",
        "",
        benchmark_table(step1, columns),
        "",
        "## Every horizon step",
        "",
        benchmark_table(rows, columns),
        "",
        "## Dynamics: can the model predict the next state at all?",
        "",
        "Next-state error in standardised units. `persistence` is the floor -- it "
        "copies the current state forward. A model that cannot beat it has not "
        "learned dynamics, whatever its classification scores say.",
        "",
        benchmark_table(dyn_rows, ["Model", "Next-state MSE", "MAE", "RMSE"]),
        "",
    ]
    if not payload.get("torch_available"):
        lines += [
            "> **Note.** PyTorch was not installed in the environment that produced "
            "this report, so the temporal transformer row is absent. Install the "
            "requirements and re-run `atdrps benchmark` to fill it in.",
            "",
        ]
    text = "\n".join(lines)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
