"""The PyTorch build-selection rule.

A driver reports the highest CUDA runtime it supports; a wheel built for an
equal or lower runtime works, a higher one does not. Choosing the wrong one is
the common way a GPU machine ends up silently training on CPU, so the rule is
pinned by tests rather than left to a comment.

Build tags verified against pytorch.org/get-started/previous-versions on
2026-09-15: 2.7.0 publishes cu118, cu126 and cu128.
"""

from __future__ import annotations

import unittest

from scripts.setup_torch import CUDA_BUILDS, build_command, choose_build


class TestChooseBuild(unittest.TestCase):
    def test_exact_matches(self):
        self.assertEqual(choose_build((12, 8)), "cu128")
        self.assertEqual(choose_build((12, 6)), "cu126")
        self.assertEqual(choose_build((11, 8)), "cu118")

    def test_newer_driver_takes_the_newest_published_build(self):
        """A 12.9 or 13.0 driver runs cu128 wheels; there is no cu130 wheel."""
        self.assertEqual(choose_build((12, 9)), "cu128")
        self.assertEqual(choose_build((13, 0)), "cu128")

    def test_driver_between_builds_falls_back_not_up(self):
        """12.7 must not be handed a cu128 wheel it cannot run."""
        self.assertEqual(choose_build((12, 7)), "cu126")
        self.assertEqual(choose_build((12, 0)), "cu118")

    def test_too_old_and_absent_drivers_use_cpu(self):
        self.assertEqual(choose_build((11, 7)), "cpu")
        self.assertEqual(choose_build(None), "cpu")

    def test_builds_are_listed_newest_first(self):
        mins = [m for _, m in CUDA_BUILDS]
        self.assertEqual(mins, sorted(mins, reverse=True),
                         "choose_build takes the first match, so order matters")


class TestBuildCommand(unittest.TestCase):
    def test_cuda_command_carries_the_matching_index_url(self):
        cmd = build_command("cu128")
        self.assertIn("--index-url", cmd)
        self.assertIn("https://download.pytorch.org/whl/cu128", cmd)

    def test_cpu_command_has_no_index_url(self):
        """The CPU build comes from PyPI; pointing at a CUDA index would be wrong."""
        self.assertNotIn("--index-url", build_command("cpu"))

    def test_pinned_by_default_and_unpinnable_on_request(self):
        self.assertTrue(any("torch==" in c for c in build_command("cpu")))
        self.assertTrue(any(c == "torch" for c in build_command("cpu", pinned=False)))


if __name__ == "__main__":
    unittest.main()
