#!/usr/bin/env python3
"""Build a windowed training corpus and cache it to disk.

Generating traffic, assembling flows and windowing is the slow part of a
training run and it is completely deterministic, so it is done once here and
cached.  ``atdrps train`` then reads the cache.

    python scripts/make_corpus.py --captures 40 --out data/corpus.npz

Every capture is a different seed, so the corpus contains campaigns that
complete and campaigns that stop early, fast scans and slow ones, and benign
bulk transfers that look like exfiltration until you look at what preceded them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atdrps.data.flows import assemble_flows                      # noqa: E402
from atdrps.data.schema import PacketTable, STAGES                # noqa: E402
from atdrps.data.synth import generate_capture                    # noqa: E402
from atdrps.data.windows import build_windows                     # noqa: E402


def _build_one(spec: tuple):
    """Generate, assemble and window exactly one capture.

    Module level and taking a plain tuple so ProcessPoolExecutor can pickle it.
    Each capture is driven by its own seed and touches no shared state, so the
    corpus is identical however many workers build it -- the seed, not the
    arrival order, decides the contents.
    """
    seed, duration_s, campaigns, intensity, window_s, history_s = spec
    cap = generate_capture(seed=seed, duration_s=duration_s, n_campaigns=campaigns,
                           background_intensity=intensity)
    flows = assemble_flows(PacketTable.from_records(cap.packets))
    states = build_windows(flows, window_size_s=window_s,
                           stage_timeline=cap.stage_at, history_s=history_s)
    return states, len(cap.packets), len(flows)


def build(captures: int, duration_s: float, campaigns: int, window_s: float,
          intensity: float, seed0: int, history_s: float, verbose: bool = True,
          workers: int | None = None):
    """Build the corpus, one capture per seed, across all available cores.

    Corpus generation is the slowest part of a training run and every capture
    is independent, so it parallelises perfectly -- this is what keeps the CPU
    busy while the GPU is idle waiting for data to train on.
    """
    from atdrps.engine.pool import PoolConfig, default_workers, map_chunks

    specs = [(seed0 + i, duration_s, campaigns, intensity, window_s, history_s)
             for i in range(captures)]
    n_workers = workers if workers is not None else default_workers(cap=16)
    cfg = PoolConfig(workers=n_workers, backend="process", min_units=2)

    t0 = time.time()
    if verbose:
        print(f"  building {captures} captures on {min(n_workers, captures)} worker(s)",
              flush=True)
    results = map_chunks(_build_one, specs, cfg, work_size=captures)

    out = []
    for i, (states, n_packets, n_flows) in enumerate(results):
        out.append(states)
        if verbose:
            print(f"  [{i + 1}/{captures}] seed={seed0 + i} "
                  f"{n_packets:>7} packets -> {n_flows:>5} flows -> "
                  f"{len(states):>4} windows", flush=True)
    if verbose:
        print(f"  corpus built in {time.time() - t0:.1f}s", flush=True)
    return out


def save(path: Path, captures) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": np.asarray(captures[0].feature_names),
        "window_size_s": np.asarray(captures[0].window_size_s),
        "n_captures": np.asarray(len(captures)),
    }
    for i, cap in enumerate(captures):
        payload[f"X_{i}"] = cap.X
        payload[f"stage_{i}"] = cap.stage
        payload[f"mask_{i}"] = cap.stage_mask
        payload[f"infil_{i}"] = cap.infiltration
        payload[f"ts_{i}"] = cap.ts_start
    np.savez_compressed(path, **payload)


def load(path: Path):
    from atdrps.data.windows import WindowedStates

    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["feature_names"]]
    window_s = float(data["window_size_s"])
    out = []
    for i in range(int(data["n_captures"])):
        out.append(WindowedStates(
            X=data[f"X_{i}"], feature_names=names, ts_start=data[f"ts_{i}"],
            window_size_s=window_s, stage=data[f"stage_{i}"],
            stage_mask=data[f"mask_{i}"], infiltration=data[f"infil_{i}"],
            label_source="ground-truth timeline",
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=int, default=40)
    ap.add_argument("--duration", type=float, default=5400.0)
    ap.add_argument("--campaigns", type=int, default=4)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--intensity", type=float, default=0.2)
    ap.add_argument("--history", type=float, default=600.0)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/corpus.npz"))
    args = ap.parse_args()

    print(f"building {args.captures} captures of {args.duration:g}s "
          f"({args.window:g}s windows)")
    started = time.time()
    captures = build(args.captures, args.duration, args.campaigns, args.window,
                     args.intensity, args.seed0, args.history)
    save(args.out, captures)

    stages = np.concatenate([c.stage[c.stage_mask] for c in captures])
    counts = {STAGES[i]: int((stages == i).sum()) for i in np.unique(stages)}
    total = int(sum(len(c) for c in captures))
    print(f"\n{total} windows across {len(captures)} captures "
          f"in {time.time() - started:.0f}s")
    print(f"stage distribution: {counts}")
    print(f"written to {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
