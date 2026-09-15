#!/usr/bin/env bash
# ============================================================================
#  ATDRPS - one-time setup  (Linux / macOS / WSL)
#
#      ./install.sh              core install
#      ./install.sh --torch      also install PyTorch (CPU build)
#      ./install.sh --torch-cuda also install PyTorch (CUDA 12.1 build)
#      ./install.sh --no-test    skip the test suite at the end
#
#  Creates .venv in this folder, installs the requirements, and proves the
#  install by running the test suite. Nothing is installed system-wide.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

usage() { sed -n '2,/^# =\{10,\}$/p' "$0" | sed -e '1d' -e '$d' -e 's/^# \{0,1\}//'; }

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; CYN=$'\033[36m'; OFF=$'\033[0m'

say()  { printf '%s\n' "${CYN}==>${OFF} ${BOLD}$*${OFF}"; }
ok()   { printf '%s\n' "  ${GRN}ok${OFF}  $*"; }
warn() { printf '%s\n' "  ${YLW}!!${OFF}  $*"; }
die()  { printf '%s\n' "  ${RED}xx${OFF}  $*" >&2; exit 1; }

WANT_TORCH=""
RUN_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --torch)      WANT_TORCH="cpu" ;;
    --torch-cuda) WANT_TORCH="cu121" ;;
    --no-test)    RUN_TESTS=0 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $arg  (try --help)" ;;
  esac
done

printf '\n%s\n' "${BOLD}  ATDRPS${OFF}  Advanced Threat Detection and Reinforcement Protocol System"
printf '%s\n\n' "${DIM}  SIH 2026 | PS 26153 | NTRO | Team Bell Labs${OFF}"

# ---------------------------------------------------------------- python
say "looking for Python 3.9+"
PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  fi
done
[ -n "$PY" ] || die "no Python 3.9+ found. Install it from https://www.python.org/downloads/"
ok "$($PY --version 2>&1)  ->  $(command -v "$PY")"

# ------------------------------------------------------------------ venv
if [ -d "$VENV" ]; then
  say "reusing the existing virtual environment"
  ok ".venv"
else
  say "creating a virtual environment in .venv"
  "$PY" -m venv "$VENV" 2>/dev/null || die \
    "could not create .venv. On Debian/Ubuntu: sudo apt install python3-venv"
  ok ".venv created"
fi

VPY="$VENV/bin/python"
[ -x "$VPY" ] || die "the virtual environment looks broken - delete .venv and re-run"

# -------------------------------------------------------------- packages
say "upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 \
  && ok "pip $("$VPY" -m pip --version | awk '{print $2}')" \
  || warn "could not upgrade pip - continuing with the bundled version"

say "installing requirements"
if "$VPY" -m pip install --quiet -r requirements.txt; then
  ok "core requirements installed"
else
  warn "pip could not install everything from requirements.txt"
  warn "if this machine is offline, see the note at the end"
fi

if [ -n "$WANT_TORCH" ]; then
  say "installing PyTorch ($WANT_TORCH build)"
  if "$VPY" -m pip install --quiet torch --index-url "https://download.pytorch.org/whl/$WANT_TORCH"; then
    ok "PyTorch installed"
  else
    warn "PyTorch install failed - ATDRPS still runs on the numpy backend"
  fi
fi

# ------------------------------------------------------------- self-check
say "checking the install"
"$VPY" - <<'PYEOF'
import importlib, sys
missing = []
for mod in ("numpy", "pandas", "scipy", "sklearn", "yaml", "flask", "matplotlib"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print("  \033[31mxx\033[0m  missing: " + ", ".join(missing))
    sys.exit(1)
print("  \033[32mok\033[0m  core packages import cleanly")
try:
    import torch
    dev = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"  \033[32mok\033[0m  PyTorch {torch.__version__} ({dev}) - transformer backend available")
except ImportError:
    print("  \033[33m!!\033[0m  PyTorch not installed - the numpy backend will be used")
    print("      to add it later:  ./install.sh --torch")
PYEOF
[ $? -eq 0 ] || die "the install is incomplete - see the messages above"

# ----------------------------------------------------------------- tests
if [ "$RUN_TESTS" -eq 1 ]; then
  say "running the test suite"
  if "$VPY" tests/run_tests.py | tail -n 4; then
    ok "tests passed"
  else
    die "tests failed - do not trust this install"
  fi
fi

printf '\n%s\n' "${GRN}${BOLD}  Setup complete.${OFF}"
printf '%s\n'   "  Next:  ${BOLD}./run.sh${OFF}          interactive menu"
printf '%s\n'   "         ${BOLD}./run.sh all${OFF}      full pipeline: corpus, benchmark, demo"
printf '%s\n'   "         ${BOLD}./run.sh dashboard${OFF}  offline dashboard"
printf '\n%s\n' "${DIM}  Offline machine? Copy a wheelhouse across and run:"
printf '%s\n\n' "    .venv/bin/pip install --no-index --find-links=wheels -r requirements.txt${OFF}"
