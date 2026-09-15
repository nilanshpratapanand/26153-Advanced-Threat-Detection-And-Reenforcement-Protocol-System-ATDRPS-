"""One command to take ATDRPS from a fresh checkout to a trained transformer.

Stages, in order:

    1. corpus   -- generate labelled traffic across every CPU core
    2. train    -- fit the temporal transformer on the GPU (CUDA if present)
    3. benchmark-- train every model and write the comparison table
    4. verify   -- run a real capture end to end through what was just trained

The split of work is deliberate: corpus generation is CPU-bound and perfectly
parallel (one capture per seed, no shared state), while training is GPU-bound.
Running them as separate stages means the machine's CPU builds the data and the
GPU trains on it, rather than one sitting idle waiting for the other.

If PyTorch is missing or has no CUDA, the transformer stage is skipped with a
clear message and the linear-dynamics backend remains the shipped model -- the
pipeline is designed so that is a degraded result, never a broken one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(args: list[str], label: str) -> int:
    print(f"\n{'=' * 68}\n  {label}\n{'=' * 68}", flush=True)
    t0 = time.time()
    rc = subprocess.call([sys.executable, "-m", *args], cwd=str(REPO))
    print(f"  -> {label}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return rc


def torch_status() -> tuple[bool, str]:
    """Is PyTorch importable, and does it actually see a GPU?"""
    try:
        import torch
    except ImportError:
        return False, "not installed"
    if torch.cuda.is_available():
        return True, f"{torch.__version__}, CUDA on {torch.cuda.get_device_name(0)}"
    return True, f"{torch.__version__}, CPU only (no CUDA device visible)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build a corpus and train ATDRPS end to end")
    ap.add_argument("--captures", type=int, default=160,
                    help="corpus size in captures (default 160; 48 is the quick option)")
    ap.add_argument("--duration", type=float, default=5400.0, help="seconds per capture")
    ap.add_argument("--campaigns", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU workers for corpus generation (default: all cores)")
    ap.add_argument("--corpus", default="data/corpus.npz")
    ap.add_argument("--skip-corpus", action="store_true",
                    help="reuse an existing corpus file")
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override training epochs")
    args = ap.parse_args(argv)

    have_torch, status = torch_status()
    print("ATDRPS -- full training run")
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  torch   : {status}")
    print(f"  corpus  : {args.captures} captures x {args.duration:.0f}s")
    if not have_torch:
        print("\n  PyTorch is not installed, so the transformer stage will be skipped.")
        print("  Run setup_torch.bat first if you want the GPU model.")

    corpus_path = REPO / args.corpus
    if args.skip_corpus:
        print(f"\n  reusing existing corpus at {args.corpus}")
    else:
        cmd = ["atdrps.cli", "corpus", "--captures", str(args.captures),
               "--duration", str(args.duration), "--campaigns", str(args.campaigns),
               "--out", args.corpus]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if _run(cmd, f"1/4  corpus  ({args.captures} captures, all CPU cores)"):
            return 1

    if have_torch and not args.skip_transformer:
        # --set is a GLOBAL flag on the top-level parser, so it has to precede
        # the subcommand; argparse rejects it afterwards.
        cmd = ["atdrps.cli"]
        if args.epochs:
            cmd += ["--set", f"train.epochs={args.epochs}"]
        cmd += ["train", "--corpus", args.corpus,
                "--backend", "torch", "--split", "group",
                "--out", "artifacts/model-transformer"]
        if _run(cmd, "2/4  train temporal transformer (GPU if available)"):
            print("\n  transformer training failed; the linear model below is still valid")
    else:
        print("\n  2/4  transformer stage skipped")

    if _run(["atdrps.cli", "benchmark", "--corpus", args.corpus, "--split", "group"],
            "3/4  benchmark every model and write the comparison table"):
        return 1

    model_dir = ("artifacts/model-transformer"
                 if (REPO / "artifacts/model-transformer/config.json").exists()
                 else "artifacts/model-linear")
    demo = REPO / "data/demo/capture.pcap"
    if demo.exists():
        _run(["atdrps.cli", "predict", str(demo), "--model", model_dir],
             f"4/4  verify end to end using {model_dir}")
    else:
        print("\n  4/4  no demo capture found; run `atdrps synth --out data/demo` to make one")

    print(f"\n{'=' * 68}")
    print("  done. artifacts/ now holds the trained model(s);")
    print("  docs/BENCHMARKS.md has the comparison table.")
    print(f"{'=' * 68}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
