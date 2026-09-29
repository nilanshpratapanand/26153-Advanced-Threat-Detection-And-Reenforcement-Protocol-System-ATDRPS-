"""Does more training data or score smoothing help?  Chosen on validation, measured once on fresh test seeds.

    python scripts/forecast_improve.py --background CAPTURE --cache STATES.pkl

Test overlays use seeds the earlier benchmark never saw (3000+), on the same held-out time
segment of the background.  The background is shared, so the test is not independent of it.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atdrps.data.overlay import load_background                        # noqa: E402
from atdrps.forecast.corpus import SEGMENTS, background_only_states, build_overlay_states  # noqa: E402
from atdrps.forecast.hazard import HazardModel, NetworkProfile         # noqa: E402
from atdrps.forecast.protocol import LATE_STAGES, build_onset_samples, evaluate  # noqa: E402


class Smoothed:
    """Mean of the model's score over the last k windows (needs context L + k - 1)."""

    def __init__(self, model, k, L):
        self.model, self.k, self.L = model, k, L

    def score(self, ctx):
        return np.mean([self.model.score(ctx[:, j:j + self.L]) for j in range(self.k)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--n-test", type=int, default=40)
    a = ap.parse_args()
    L, K, KMAX = a.context, a.horizon, 4
    t0 = time.time()
    bg = load_background(a.background)
    segs = {k: bg.segment(*v) for k, v in SEGMENTS.items()}
    if Path(a.cache).exists():
        st = pickle.loads(Path(a.cache).read_bytes())
    else:
        st = {"train": build_overlay_states(segs["train"], range(a.n_train), n_campaigns=1),
              "val": build_overlay_states(segs["val"], range(1000, 1000 + a.n_val), n_campaigns=1),
              "test": build_overlay_states(segs["test"], range(3000, 3000 + a.n_test), n_campaigns=1),
              "bg_train": background_only_states(segs["train"])}
        Path(a.cache).write_bytes(pickle.dumps(st))
    print(f"states ready [{time.time() - t0:.0f}s]", flush=True)
    profile = NetworkProfile().fit(st["bg_train"].X)
    mk = lambda key, ctx: build_onset_samples(st[key], context=ctx, horizon=K, positive_stages=LATE_STAGES)
    val_long, test_long = mk("val", L + KMAX - 1), mk("test", L + KMAX - 1)
    print("val ", val_long.describe()); print("test", test_long.describe(), flush=True)

    rows = []
    for n_train in (60, 120, a.n_train):
        train = build_onset_samples(st["train"][:n_train], context=L, horizon=K, positive_stages=LATE_STAGES)
        mid = HazardModel(profile).fit(train, None)
        for k in (1, 2, 3, 4):
            sm = Smoothed(mid, k, L)
            # windows are aligned on the newest end, so a longer context still ends at t
            vs, ts_ = sm.score(val_long.context), sm.score(test_long.context)
            rv = evaluate(val_long, vs, fpr_budget=0.05, n_boot=0)
            rt = evaluate(test_long, ts_, val_scores=vs, val_samples=val_long, fpr_budget=0.05, n_boot=200)
            rows.append((n_train, k, rv, rt))
            print(f"n_train={n_train:3d} smooth={k}  VAL auc={rv['auc']:.3f} ap={rv['ap']:.3f}   "
                  f"TEST auc={rt['auc']:.3f} [{rt['ci95']['auc'][0]:.3f},{rt['ci95']['auc'][1]:.3f}] ap={rt['ap']:.3f} "
                  f"@val-thr(5%FPR): tpr={rt['operating']['tpr']:.3f} fpr={rt['operating']['fpr']:.3f} "
                  f"event-recall={rt['operating']['event_recall']:.2f} ({rt['operating']['n_events']} events) "
                  f"lead={rt['operating']['median_lead_s']:.0f}s  [{time.time() - t0:.0f}s]", flush=True)
    best = max(rows, key=lambda r: r[2]["auc"])
    print(f"\nchosen on VALIDATION AUC: n_train={best[0]} smooth={best[1]} -> TEST auc {best[3]['auc']:.3f}")


if __name__ == "__main__":
    main()
