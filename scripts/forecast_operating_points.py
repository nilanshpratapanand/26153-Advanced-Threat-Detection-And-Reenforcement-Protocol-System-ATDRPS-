"""Recall against false-alarm budget for the configuration chosen on validation
(240 training overlays, 3-window score smoothing).  Thresholds come from validation only.

    python scripts/forecast_operating_points.py --background CAPTURE --cache STATES.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import beta                                                    # noqa: E402

from atdrps.data.overlay import load_background                                # noqa: E402
from atdrps.forecast.corpus import SEGMENTS, background_only_states            # noqa: E402
from atdrps.forecast.hazard import HazardModel, NetworkProfile                 # noqa: E402
from atdrps.forecast.protocol import (LATE_STAGES, build_onset_samples,        # noqa: E402
                                      evaluate, threshold_for_fpr)

L, K, SMOOTH = 16, 5, 3


def smoothed(model, ctx):
    return np.mean([model.score(ctx[:, j:j + L]) for j in range(SMOOTH)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    st = pickle.loads(Path(a.cache).read_bytes())
    bg = load_background(a.background)
    test_bg = background_only_states(bg.segment(*SEGMENTS["test"]))
    profile = NetworkProfile().fit(st["bg_train"].X)
    train = build_onset_samples(st["train"], context=L, horizon=K, positive_stages=LATE_STAGES)
    model = HazardModel(profile).fit(train, None)
    val = build_onset_samples(st["val"], context=L + SMOOTH - 1, horizon=K, positive_stages=LATE_STAGES)
    test = build_onset_samples(st["test"], context=L + SMOOTH - 1, horizon=K, positive_stages=LATE_STAGES)
    vs, ts_ = smoothed(model, val.context), smoothed(model, test.context)
    X = np.asarray(test_bg.X, dtype=np.float32)
    bg_ctx = np.stack([X[t - (L + SMOOTH - 1) + 1:t + 1] for t in range(L + SMOOTH - 2, len(X))])
    bs = smoothed(model, bg_ctx)
    print(f"test: {test.describe()}\nvalidation negatives: {int((val.onset_in_k == 0).sum())}   "
          f"clean real-background test windows: {len(bs)}\n")
    print(f"{'budget FPR':>10s} {'val thr':>8s} | {'test FPR':>8s} {'test TPR':>8s} {'event recall':>13s} {'lead s':>7s} "
          f"{'FA/h on test neg.':>18s} | {'real-bg FA/h (95% upper)':>26s} | {'PPV @1e-3':>9s}")
    for b in (0.005, 0.01, 0.02, 0.05):
        thr = threshold_for_fpr(vs, val.onset_in_k.astype(int), b)
        r = evaluate(test, ts_, threshold=thr, n_boot=0)
        o = r["operating"]
        k, n = int((bs >= thr).sum()), len(bs)
        up = float(beta.ppf(0.975, k + 1, n - k)) * 120
        print(f"{b:10.3f} {thr:8.3f} | {o['fpr']:8.3f} {o['tpr']:8.3f} {o['event_recall']:5.2f} ({round(o['event_recall'] * o['n_events'])}/{o['n_events']}) "
              f"{o['median_lead_s']:7.0f} {o['fpr'] * 120:18.1f} | {k / n * 120:8.1f} ({up:6.1f})           | {o['ppv@0.001']:9.4f}")
    print("\n(FA/h = false alerts per hour at 30 s windows; 'PPV @1e-3' = precision if one window in a thousand is an escalation onset)")


if __name__ == "__main__":
    main()
