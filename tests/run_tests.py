#!/usr/bin/env python3
"""Zero-dependency test runner for ATDRPS.

Uses only ``unittest`` from the standard library, so the suite runs in an
air-gapped environment with nothing installed beyond the core requirements
(no pytest).  Tests that need PyTorch skip themselves when torch is absent.

    python tests/run_tests.py            # everything
    python tests/run_tests.py test_pcap  # one module
    python tests/run_tests.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str]) -> int:
    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    names = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if names:
        suite = unittest.TestSuite()
        for name in names:
            mod = name if name.startswith("tests.") else f"tests.{name}"
            suite.addTests(loader.loadTestsFromName(mod))
    else:
        suite = loader.discover(str(TESTS_DIR), pattern="test_*.py", top_level_dir=str(REPO_ROOT))

    result = unittest.TextTestRunner(verbosity=verbosity, buffer=False).run(suite)
    print(
        f"\n[atdrps] ran={result.testsRun} "
        f"failures={len(result.failures)} errors={len(result.errors)} "
        f"skipped={len(result.skipped)}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
