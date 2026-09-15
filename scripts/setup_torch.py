"""Work out the right PyTorch install command for this machine, and run it.

Why this exists rather than "just pip install torch": the wheel you need
depends on the CUDA runtime your *driver* supports, and getting it wrong is the
single most common way a GPU box ends up silently training on CPU. pip install
torch with no index URL gives you the CPU build on Windows, which works and is
slow, and nothing warns you.

Sources (verified 2026-09-15):
  https://pytorch.org/get-started/locally/
  https://pytorch.org/get-started/previous-versions/

PyTorch 2.7.0 publishes Windows/Linux CUDA wheels for cu118, cu126 and cu128;
2.8.0 adds cu129. Python 3.10-3.14 are supported.

Pairing rule: a driver reports the highest CUDA runtime it supports (nvidia-smi
"CUDA Version"). A wheel built for an equal or lower CUDA runtime works; a
higher one does not. So pick the newest published build <= the driver's number.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

__all__ = ["detect_driver_cuda", "choose_build", "build_command", "CUDA_BUILDS"]

# published CUDA builds, newest first -- (tag, minimum driver CUDA runtime)
CUDA_BUILDS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("cu128", (12, 8)),
    ("cu126", (12, 6)),
    ("cu118", (11, 8)),
)

TORCH_VERSION = "2.7.0"
TORCHVISION_VERSION = "0.22.0"
TORCHAUDIO_VERSION = "2.7.0"


def detect_driver_cuda() -> tuple[int, int] | None:
    """Highest CUDA runtime this NVIDIA driver supports, from nvidia-smi.

    Returns ``None`` when there is no NVIDIA GPU or no driver -- which is a
    normal outcome, not an error: the caller then installs the CPU build.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def choose_build(driver_cuda: tuple[int, int] | None) -> str:
    """Newest published CUDA build the driver can actually run, else CPU."""
    if driver_cuda is None:
        return "cpu"
    for tag, minimum in CUDA_BUILDS:
        if driver_cuda >= minimum:
            return tag
    return "cpu"


def build_command(tag: str, pinned: bool = True) -> list[str]:
    """The exact pip command for a build tag."""
    if pinned:
        pkgs = [f"torch=={TORCH_VERSION}",
                f"torchvision=={TORCHVISION_VERSION}",
                f"torchaudio=={TORCHAUDIO_VERSION}"]
    else:
        pkgs = ["torch", "torchvision", "torchaudio"]
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    if tag != "cpu":
        cmd += ["--index-url", f"https://download.pytorch.org/whl/{tag}"]
    return cmd


def verify() -> int:
    """Report what actually got installed -- CPU vs CUDA, and which device."""
    try:
        import torch
    except ImportError:
        print("  torch is still not importable -- the install did not succeed")
        return 1
    print(f"  torch                : {torch.__version__}")
    print(f"  built with CUDA      : {torch.version.cuda or 'no (CPU-only build)'}")
    available = torch.cuda.is_available()
    print(f"  CUDA available now   : {available}")
    if available:
        print(f"  device               : {torch.cuda.get_device_name(0)}")
    else:
        print("  -> training will run on CPU. That works, it is just slower.")
    x = torch.rand(5, 3)
    print(f"  smoke test           : tensor {tuple(x.shape)} OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="install the right PyTorch for this machine")
    ap.add_argument("--dry-run", action="store_true", help="print the command, install nothing")
    ap.add_argument("--cpu", action="store_true", help="force the CPU build")
    ap.add_argument("--verify-only", action="store_true", help="only report what is installed")
    ap.add_argument("--latest", action="store_true", help="unpinned versions instead of 2.7.0")
    args = ap.parse_args(argv)

    print("ATDRPS -- PyTorch setup")
    print(f"  python               : {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  ! PyTorch 2.7 needs Python 3.10 or newer. Upgrade Python first.")
        return 2

    if args.verify_only:
        return verify()

    driver = None if args.cpu else detect_driver_cuda()
    if driver:
        print(f"  NVIDIA driver CUDA   : {driver[0]}.{driver[1]}")
    else:
        print("  NVIDIA driver CUDA   : none detected")
    tag = "cpu" if args.cpu else choose_build(driver)
    print(f"  chosen build         : {tag}")

    cmd = build_command(tag, pinned=not args.latest)
    print("  command              :")
    print("    " + " ".join(cmd))
    if args.dry_run:
        return 0
    print()
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\n  pip exited {rc}. Nothing was changed by this script itself.")
        return rc
    print("\nverifying:")
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
