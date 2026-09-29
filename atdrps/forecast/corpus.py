"""Build labelled window sequences from a real background plus injected campaigns.

Train / validation / test come from *disjoint time segments of the background*, so no
background window appears in more than one split.  Even so, every overlay of one segment
shares that segment's background; the number of independent networks is still one, and
results must be read that way.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from ..data.flows import assemble_flows
from ..data.overlay import Background, overlay_campaigns
from ..data.schema import PacketTable
from ..data.windows import WindowedStates, build_windows

__all__ = ["build_overlay_states", "background_only_states", "SEGMENTS"]

SEGMENTS = {"train": (0.00, 0.55), "val": (0.55, 0.72), "test": (0.72, 1.00)}

_BG: Background | None = None          # set before forking so workers share it copy-on-write


def _states(records, stage_timeline, window_s, history_s):
    flows = assemble_flows(PacketTable.from_records(records))
    return build_windows(flows, window_size_s=window_s, stage_timeline=stage_timeline,
                         history_s=history_s)


def _one(spec):
    seed, kwargs, window_s, history_s = spec
    cap = overlay_campaigns(_BG, seed, **kwargs)
    return _states(cap.packets, cap.stage_at, window_s, history_s)


def build_overlay_states(bg: Background, seeds, *, window_s: float = 30.0,
                         history_s: float = 600.0, workers: int | None = None,
                         **overlay_kwargs) -> list[WindowedStates]:
    """One :class:`WindowedStates` per seed (each a different campaign placement)."""
    global _BG
    _BG = bg
    specs = [(int(s), overlay_kwargs, window_s, history_s) for s in seeds]
    workers = workers or min(4, os.cpu_count() or 1)
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_one, specs, chunksize=1)
    return [_one(s) for s in specs]


def background_only_states(bg: Background, *, window_s: float = 30.0,
                           history_s: float = 600.0) -> WindowedStates:
    """The untouched background, every window labelled benign (used to measure false alarms)."""
    return _states(bg.records, lambda ts: "Benign", window_s, history_s)
