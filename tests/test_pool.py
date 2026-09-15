"""The worker pool, and the property that makes sharded flow assembly safe.

Parallelism is only worth anything if the answer does not change. Flow
assembly shards on the canonical bidirectional conversation id precisely so
that every packet of a conversation lands in one shard -- sharding on time
would cut long flows in half and invent extra ones. These tests assert the
equivalence rather than trusting the argument.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from atdrps.data.flows import assemble_flows, assemble_flows_parallel
from atdrps.data.schema import PacketTable
from atdrps.data.synth import generate_capture
from atdrps.engine.pool import PoolConfig, default_workers, map_chunks


def _square(x: int) -> int:        # module level: ProcessPoolExecutor pickles it
    return x * x


class TestPoolConfig(unittest.TestCase):
    def test_default_workers_is_sane(self):
        n = default_workers()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, 8)

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            PoolConfig(backend="magic")

    def test_small_workloads_stay_serial(self):
        cfg = PoolConfig(workers=4, backend="process", min_units=1000)
        self.assertEqual(cfg.effective_backend(10), "serial")
        self.assertEqual(cfg.effective_backend(5000), "process")

    def test_single_worker_is_always_serial(self):
        self.assertEqual(PoolConfig(workers=1, backend="process").effective_backend(10**9),
                         "serial")


class TestMapChunks(unittest.TestCase):
    def test_results_keep_input_order(self):
        units = list(range(12))
        for backend in ("serial", "thread", "process"):
            cfg = PoolConfig(workers=3, backend=backend, min_units=0)
            self.assertEqual(map_chunks(_square, units, cfg), [u * u for u in units],
                             f"{backend} backend changed order or values")

    def test_work_size_not_unit_count_decides_the_backend(self):
        """The bug this guards: gating on len(units) sent four big chunks of
        twenty thousand packets down the serial path, producing a 1.02x
        "speedup" that looked like parallelism simply not helping."""
        cfg = PoolConfig(workers=2, backend="thread", min_units=1000)
        # four units, but they represent a lot of work
        self.assertEqual(cfg.effective_backend(4), "serial")
        self.assertEqual(cfg.effective_backend(80_000), "thread")
        # map_chunks must honour the declared work size
        self.assertEqual(map_chunks(_square, [1, 2, 3, 4], cfg, work_size=80_000),
                         [1, 4, 9, 16])

    def test_empty_input(self):
        self.assertEqual(map_chunks(_square, [], PoolConfig(min_units=0)), [])


class TestShardedFlowAssembly(unittest.TestCase):
    """The equivalence that makes the parallel path legitimate."""

    @classmethod
    def setUpClass(cls):
        cap = generate_capture(seed=99, duration_s=600, n_campaigns=1,
                               background_intensity=0.25)
        cls.table = PacketTable.from_records(cap.packets).sort_by_time()
        cls.serial = assemble_flows(cls.table)

    def _assert_same(self, other: pd.DataFrame, label: str):
        self.assertEqual(len(other), len(self.serial), f"{label}: flow count differs")
        key = ["start_ts", "src_ip", "dst_ip", "dst_port_raw"]
        a = self.serial.sort_values(key).reset_index(drop=True)
        b = other.sort_values(key).reset_index(drop=True)
        numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
        pd.testing.assert_frame_equal(
            a[numeric].round(6), b[numeric].round(6),
            check_dtype=False, obj=f"{label} vs serial",
        )

    def test_thread_sharding_is_identical_to_serial(self):
        cfg = PoolConfig(workers=4, backend="thread", min_units=1)
        self._assert_same(assemble_flows_parallel(self.table, config=cfg), "thread")

    def test_process_sharding_is_identical_to_serial(self):
        cfg = PoolConfig(workers=4, backend="process", min_units=1)
        self._assert_same(assemble_flows_parallel(self.table, config=cfg), "process")

    def test_shard_count_does_not_change_the_answer(self):
        """A flow must never straddle a shard boundary at any shard count."""
        for workers in (2, 3, 5, 8):
            cfg = PoolConfig(workers=workers, backend="thread", min_units=1)
            self._assert_same(assemble_flows_parallel(self.table, config=cfg),
                              f"{workers} shards")

    def test_output_is_sorted_by_start_time(self):
        cfg = PoolConfig(workers=4, backend="thread", min_units=1)
        out = assemble_flows_parallel(self.table, config=cfg)
        self.assertTrue((np.diff(out["start_ts"].to_numpy()) >= 0).all(),
                        "reassembled shards must come back in time order")

    def test_empty_capture_returns_the_empty_schema(self):
        cfg = PoolConfig(workers=4, backend="thread", min_units=1)
        out = assemble_flows_parallel(PacketTable.empty(0), config=cfg)
        self.assertEqual(len(out), 0)
        self.assertEqual(list(out.columns), list(self.serial.columns))


if __name__ == "__main__":
    unittest.main()
