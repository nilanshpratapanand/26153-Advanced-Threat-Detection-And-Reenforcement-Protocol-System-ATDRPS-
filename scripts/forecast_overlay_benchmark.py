"""Benchmark forecasters on a real background capture with injected, labelled campaigns.

    python scripts/forecast_overlay_benchmark.py --background my_capture.pcapng --out report.json

Train / validation / test are disjoint time segments of the background.  Thresholds are chosen
on validation overlays at ``--budget`` false-positive rate.  The pure background test segment
(no injection, assumed benign) gives the false alerts per hour a deployment would see.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atdrps.data.overlay import load_background                     # noqa: E402
from atdrps.forecast.baselines import (                             # noqa: E402
    CusumBaseline, IsolationForestBaseline, MahalanobisBaseline, PriorBaseline, V1Baseline)
from atdrps.forecast.corpus import SEGMENTS, background_only_states, build_overlay_states  # noqa: E402
from atdrps.forecast.hazard import HazardModel, NetworkProfile      # noqa: E402
from atdrps.forecast.protocol import (                              # noqa: E402
    LATE_STAGES, build_onset_samples, contexts_from_states, evaluate, threshold_for_fpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=30)
    ap.add_argument("--task", choices=["onset", "escalation"], default="onset",
                    help="onset: will infiltration begin from quiet?  escalation: will compromise "
                         "spread (lateral movement/C2/exfiltration/impact) given early-stage activity?")
    ap.add_argument("--cache", default=None, help="pickle file for the built window sequences")
    ap.add_argument("--skip-v1", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    bg = load_background(a.background)
    print(f"background: {len(bg.records)} IPv4 packets, {(bg.t1 - bg.t0) / 60:.0f} min, "
          f"{len(bg.peers)} peer hosts")
    segs = {k: bg.segment(*v) for k, v in SEGMENTS.items()}
    if a.cache and Path(a.cache).exists():
        states, bgonly = pickle.loads(Path(a.cache).read_bytes())
        print(f"loaded window sequences from {a.cache}")
    else:
        states, bgonly = {}, {}
        for name, n, base in (("train", a.n_train, 0), ("val", a.n_val, 1000), ("test", a.n_test, 2000)):
            states[name] = build_overlay_states(segs[name], range(base, base + n), window_s=a.window,
                                                n_campaigns=1)
            bgonly[name] = background_only_states(segs[name], window_s=a.window)
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps((states, bgonly)))
    positive = LATE_STAGES if a.task == "escalation" else None
    data = {}
    for name in ("train", "val", "test"):
        data[name] = build_onset_samples(states[name], context=a.context, horizon=a.horizon,
                                         positive_stages=positive)
        print(f"  {name} [{a.task}]: {len(states[name])} overlays, {data[name].describe()}   "
              f"[{time.time() - t0:.0f}s]")

    profile = NetworkProfile().fit(bgonly["train"].X)
    print(f"profile fitted on {profile.n_windows} benign training-segment windows")
    models = [PriorBaseline(), CusumBaseline(), MahalanobisBaseline(), IsolationForestBaseline()]
    for m in models:
        m.fit(data["train"])
    haz = HazardModel(profile).fit(data["train"], data["val"])
    models.append(haz)
    if not a.skip_v1:
        print("training v1 world model on the same overlay training captures ...")
        models.append(V1Baseline(a.context, a.horizon).fit_captures(states["train"]))

    bg_ctx = contexts_from_states(bgonly["test"], a.context)
    rows = []
    for m in models:
        vs, ts_, bs = m.score(data["val"].context), m.score(data["test"].context), m.score(bg_ctx)
        r = evaluate(data["test"], ts_, val_scores=vs, val_samples=data["val"],
                     fpr_budget=a.budget, n_boot=200)
        thr = r["operating"]["threshold"]
        r["real_background_false_alert_rate"] = float((bs >= thr).mean()) if len(bs) else float("nan")
        r["real_background_false_alerts_per_hour"] = r["real_background_false_alert_rate"] * 3600 / a.window
        r["real_background_windows"] = int(len(bs))
        k, n_bg = int((bs >= thr).sum()), int(len(bs))
        from scipy.stats import beta
        upper = float(beta.ppf(0.975, k + 1, n_bg - k)) if 0 < n_bg and k < n_bg else float("nan")
        r["real_background_false_alerts_per_hour_upper95"] = upper * 3600 / a.window
        rows.append((m.name, r))

    print(f"\ntask={a.task}   test: {data['test'].describe()}")
    print(f"real-background test windows (assumed benign): {len(bg_ctx)}\n")
    hdr = (f"{'model':40s} {'AUC [95% CI]':22s} {'AP':>6s} {'TPR@val-thr':>11s} {'FPR':>6s} "
           f"{'evt recall':>10s} {'lead s':>7s} {'PPV@1e-3':>9s} {'real-bg FA/h (95% upper)':>26s}")
    print(hdr)
    for n, r in rows:
        o, (lo, hi) = r["operating"], r["ci95"]["auc"]
        print(f"{n:40s} {r['auc']:.3f} [{lo:.3f},{hi:.3f}]  {r['ap']:6.3f} {o['tpr']:11.3f} {o['fpr']:6.3f} "
              f"{o['event_recall']:10.3f} {o['median_lead_s']:7.0f} {o['ppv@0.001']:9.4f} "
              f"{r['real_background_false_alerts_per_hour']:8.1f} ({r['real_background_false_alerts_per_hour_upper95']:5.1f})")
    if a.out:
        Path(a.out).write_text(json.dumps({n: r for n, r in rows}, indent=2, default=float))
        print("\nwrote", a.out)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
