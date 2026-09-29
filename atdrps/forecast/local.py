"""Train and run a forecaster for *one* network, from that network's own benign capture.

The v1 model was trained on simulation and treated a benign laptop capture as an attack in
most windows.  The remedy is not a bigger model but a local one: learn what normal looks
like *here*, train on that real background with labelled attacks injected on top, and
calibrate the alert threshold on this network's own benign windows.

Two things this module refuses to hide:

* the capture you give it is **assumed benign**.  If it contains an intrusion, that intrusion
  becomes part of "normal";
* the requested false-alarm budget can only be *verified* with enough benign data.  The
  threshold is always the empirical quantile for the budget, but the exact 95 % upper bound on
  the false-alarm rate that the calibration data supports is stored next to it, and the model
  says "not verified" when that bound exceeds the budget, instead of pretending.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from ..data.overlay import load_background
from .corpus import background_only_states, build_overlay_states
from .hazard import HazardModel, NetworkProfile
from .protocol import (LATE_STAGES, build_onset_samples, contexts_from_states,
                       threshold_for_fpr)

__all__ = ["train_local", "analyse_capture", "MIN_BENIGN_MINUTES"]

MIN_BENIGN_MINUTES = 20.0
DISCLAIMER = ("An alert means 'activity like the early stages of a spreading intrusion is "
              "unusually likely, compared with this network's normal'. It is not proof of an "
              "attack, and false alerts are expected.")


def train_local(benign_path, out_dir, *, task: str = "escalation", budget_per_hour: float = 1.0,
                window_s: float = 30.0, context: int = 16, horizon: int = 5, n_train: int = 60,
                n_val: int = 24, workers: int | None = None, min_train_positives: int = 10,
                min_val_positives: int = 3, log=print) -> dict:
    if task not in ("escalation", "onset"):
        raise ValueError("task must be 'escalation' or 'onset'")
    bg = load_background(benign_path)
    minutes = (bg.t1 - bg.t0) / 60.0
    if minutes < MIN_BENIGN_MINUTES:
        raise ValueError(f"the benign capture covers {minutes:.1f} minutes; at least "
                         f"{MIN_BENIGN_MINUTES:.0f} are needed to learn a network's normal and "
                         "hold some back for calibration. Capture longer (a full day is much better).")
    train_bg, val_bg = bg.segment(0.0, 0.6), bg.segment(0.6, 1.0)
    log(f"benign capture: {minutes:.0f} min, {len(bg.records)} IPv4 packets "
        f"(assumed benign; ARP and IPv6 are not used)")

    positive = LATE_STAGES if task == "escalation" else None
    log(f"building {n_train} + {n_val} labelled overlays on your real traffic ...")
    tr_states = build_overlay_states(train_bg, range(n_train), window_s=window_s, workers=workers,
                                     n_campaigns=1)
    va_states = build_overlay_states(val_bg, range(1000, 1000 + n_val), window_s=window_s,
                                     workers=workers, n_campaigns=1)
    bg_train = background_only_states(train_bg, window_s=window_s)
    bg_val = background_only_states(val_bg, window_s=window_s)
    profile = NetworkProfile().fit(bg_train.X)

    train = build_onset_samples(tr_states, context=context, horizon=horizon, positive_stages=positive)
    val = build_onset_samples(va_states, context=context, horizon=horizon, positive_stages=positive)
    if train.n_positive < min_train_positives or val.n_positive < min_val_positives:
        raise ValueError(f"too few positive examples (train {train.n_positive}, val {val.n_positive}); "
                         "raise n_train/n_val or use a longer capture")
    log(f"  train: {train.describe()}")
    model = HazardModel(profile).fit(train, val)

    # calibration negatives: everything on the held-back segment that is not a positive
    neg_scores = np.concatenate([
        model.score(val.context[val.onset_in_k == 0]),
        model.score(contexts_from_states(bg_val, context)),
    ])
    n_cal = int(len(neg_scores))
    per_window = budget_per_hour * window_s / 3600.0
    threshold = threshold_for_fpr(neg_scores, np.zeros(n_cal, dtype=int), per_window)
    k = int((neg_scores >= threshold).sum())
    from scipy.stats import beta
    upper_rate = float(beta.ppf(0.975, k + 1, n_cal - k)) if k < n_cal else 1.0
    upper_per_hour = upper_rate * 3600.0 / window_s
    verified = upper_per_hour <= budget_per_hour
    import sklearn
    meta = {
        "task": task, "window_s": window_s, "context": context, "horizon": horizon,
        "threshold": float(threshold), "budget_per_hour": budget_per_hour,
        "calibration_windows": n_cal, "calibration_false_alerts": k,
        "false_alerts_per_hour_upper95": round(upper_per_hour, 2),
        "budget_verified": bool(verified), "benign_minutes": round(minutes, 1),
        "n_train_positive": train.n_positive, "n_val_positive": val.n_positive,
        "created_unix": int(time.time()), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "python": platform.python_version(),
        "assumption": "the training capture was benign",
        "disclaimer": DISCLAIMER,
    }
    out = Path(out_dir)
    model.save(out)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not verified:
        log(f"  NOTE: {n_cal} benign calibration windows show {k} false alert(s) at this threshold, "
            f"which only supports 'at most {upper_per_hour:.1f} false alerts/hour (95%)', not the "
            f"requested {budget_per_hour}/hour. Capture more benign traffic to verify the rate.")
    log(f"saved to {out}   threshold={threshold:.4f}   budget verified: {verified}")
    return meta


def analyse_capture(model_dir, capture_path) -> dict:
    from ..engine.inference import load_capture
    import sklearn
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    model = HazardModel.load(d)
    warnings = []
    if meta.get("sklearn") != sklearn.__version__:
        warnings.append(f"model was trained with scikit-learn {meta.get('sklearn')} but this is "
                        f"{sklearn.__version__}; retrain if results look wrong")
    if not meta.get("budget_verified"):
        warnings.append(f"the false-alarm rate is not verified: the benign calibration data only "
                        f"supports at most {meta.get('false_alerts_per_hour_upper95')} false alerts/hour "
                        f"(95%), against a requested {meta.get('budget_per_hour')}/hour")

    states, flows, kind = load_capture(capture_path, meta["window_s"], history_s=600.0)
    L = int(meta["context"])
    result = {"source_kind": kind, "n_windows": len(states), "n_flows": len(flows),
              "windows": [], "episodes": [], "warnings": warnings, "meta": meta,
              "disclaimer": DISCLAIMER}
    if len(states) < L:
        warnings.append(f"capture has {len(states)} windows but the model needs {L} "
                        f"({L * meta['window_s'] / 60:.0f} min) of context")
        return result
    ctx = contexts_from_states(states, L)
    scores, probs = model.score(ctx), model.probability(ctx)
    z = model.profile.transform(states.X)
    drift = float(np.median(np.abs(z)))
    result["drift_median_abs_z"] = drift
    if drift > 3.0:
        warnings.append(f"this traffic is far from the network the model was calibrated on "
                        f"(median |z| {drift:.1f}); scores are not meaningful here")
    thr = float(meta["threshold"])
    for i, (sc, pr) in enumerate(zip(scores, probs)):
        w = i + L - 1
        result["windows"].append({"window": int(w), "start": float(states.ts_start[w]),
                                  "score": float(sc), "probability": float(pr),
                                  "alert": bool(sc >= thr)})
    run = None
    for row in result["windows"] + [None]:
        if row is not None and row["alert"]:
            run = run or {"start_window": row["window"], "start": row["start"], "peak": row["score"], "n": 0}
            run["end_window"], run["n"], run["peak"] = row["window"], run["n"] + 1, max(run["peak"], row["score"])
        elif run is not None:
            result["episodes"].append(run)
            run = None
    return result
