"""A worker pool for the parts of the pipeline that are genuinely independent.

Measured first, then parallelised. On an 88,767-packet capture (4,737 flows,
80 windows) the single-threaded pipeline divides up like this:

    pcap parse          1.41 s   21 %
    flow assembly       2.66 s   40 %   <-- the bottleneck
    windowing + all
      family detectors  0.87 s   13 %
    scoring             0.02 s    0 %
    SHAP explanation    0.37 s    6 %

So the obvious target -- running the per-attack-family detectors concurrently
-- is chasing 13 % of the wall clock, while flow assembly next door is three
times larger. This module parallelises by measurement rather than by intuition.

Why flow assembly shards safely
-------------------------------
``assemble_flows`` groups packets by a canonical bidirectional 5-tuple id, and
every group is built independently of every other. Partitioning packets by
``key % n_shards`` therefore puts *all* packets of a conversation in exactly one
shard: no flow can straddle a boundary, no packet is counted twice, and idle and
active timeouts still see the complete packet sequence for their conversation.
Sharding on time would not have that property -- it would cut long flows in half
and invent extra ones -- which is why the partition is on the key, not the clock.

The result is deterministic: shards are reassembled and re-sorted by flow start
time, so a parallel run and a serial run produce the same table. That is
asserted in the tests, not assumed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

__all__ = ["default_workers", "map_chunks", "PoolConfig"]

T = TypeVar("T")
R = TypeVar("R")


def default_workers(cap: int = 8) -> int:
    """A sensible worker count: never more than the machine has, never wild.

    Capped because the stages being parallelised are memory-bandwidth bound as
    much as CPU bound, and because a laptop running a live capture at the same
    time should keep a core free for the capture thread.
    """
    try:
        n = len(os.sched_getaffinity(0))          # respects cgroup/taskset limits
    except AttributeError:                         # not available on Windows
        n = os.cpu_count() or 1
    return max(1, min(int(n), cap))


class PoolConfig:
    """How to run independent work units.

    ``backend`` is one of ``"serial"``, ``"thread"`` or ``"process"``.  The
    default is chosen per call site from what was actually measured, not from a
    general preference -- see the benchmark in ``docs/BENCHMARKS.md``.
    """

    __slots__ = ("workers", "backend", "min_units")

    def __init__(self, workers: int | None = None, backend: str = "process",
                 min_units: int = 2000) -> None:
        if backend not in ("serial", "thread", "process"):
            raise ValueError(f"unknown backend {backend!r}")
        self.workers = int(workers) if workers else default_workers()
        self.backend = backend
        # below this much work the pool costs more than it saves; measured, not guessed
        self.min_units = int(min_units)

    def effective_backend(self, n_units: int) -> str:
        if self.workers <= 1 or n_units < self.min_units:
            return "serial"
        return self.backend

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PoolConfig(workers={self.workers}, backend={self.backend!r})"


def _process_context():
    """Prefer ``fork`` where the platform has it.

    A spawned worker re-imports pandas, numpy and sklearn before it can do any
    work, which on a measured run cost enough to turn a 1.82x speedup into
    1.41x on the first call. ``fork`` inherits the already-imported parent.
    Windows has no fork, so there the default context is used and the pool is
    simply worth less on short inputs -- which ``min_units`` already guards.
    """
    try:
        if "fork" in mp.get_all_start_methods():
            return mp.get_context("fork")
    except (AttributeError, ValueError):
        pass
    return None


def map_chunks(fn: Callable[[T], R], units: Sequence[T],
               config: PoolConfig | None = None,
               work_size: int | None = None) -> list[R]:
    """Apply ``fn`` to every unit, in order, possibly in parallel.

    ``work_size`` is how much actual work these units represent -- packets, say
    -- as opposed to how many units there are. The two are very different: four
    chunks of twenty thousand packets each is a lot of work in very few units,
    and gating on ``len(units)`` sent exactly that case down the serial path and
    silently produced a 1.02x "speedup" that looked like the parallelism simply
    not helping. Callers that know their real workload should pass it.

    Order of results always matches order of ``units`` regardless of backend, so
    a caller can rely on the output being deterministic.
    """
    config = config or PoolConfig()
    units = list(units)
    if not units:
        return []
    gate = len(units) if work_size is None else int(work_size)
    backend = config.effective_backend(gate)
    if backend == "serial" or len(units) == 1:
        return [fn(u) for u in units]

    workers = min(config.workers, len(units))
    try:
        if backend == "thread":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(fn, units))
        ctx = _process_context()
        kwargs = {"max_workers": workers}
        if ctx is not None:
            kwargs["mp_context"] = ctx
        with ProcessPoolExecutor(**kwargs) as pool:
            return list(pool.map(fn, units))
    except (OSError, RuntimeError, ValueError):
        # a sandbox with no /dev/shm, a frozen exe without a spawn guard, a
        # restricted container -- falling back is always better than failing
        return [fn(u) for u in units]
