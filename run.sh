#!/usr/bin/env bash
# ============================================================================
#  ATDRPS - run everything  (Linux / macOS / WSL)
#
#      ./run.sh                 interactive menu
#      ./run.sh all             full pipeline: demo capture, corpus, benchmark
#      ./run.sh demo            generate a labelled synthetic capture
#      ./run.sh corpus          build the training corpus
#      ./run.sh train           train the world model
#      ./run.sh benchmark       train everything and write docs/BENCHMARKS.md
#      ./run.sh predict [file]  forecast from a capture
#      ./run.sh dashboard       start the offline dashboard
#      ./run.sh test            run the test suite
#      ./run.sh clean           remove generated data, models and reports
#
#  Everything runs locally. No cloud APIs, no telemetry, no downloads.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

usage() { sed -n '2,/^# =\{10,\}$/p' "$0" | sed -e '1d' -e '$d' -e 's/^# \{0,1\}//'; }
case "${1:-}" in -h|--help|help) usage; exit 0 ;; esac

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; CYN=$'\033[36m'; OFF=$'\033[0m'

say()  { printf '\n%s\n' "${CYN}==>${OFF} ${BOLD}$*${OFF}"; }
ok()   { printf '%s\n' "  ${GRN}ok${OFF}  $*"; }
warn() { printf '%s\n' "  ${YLW}!!${OFF}  $*"; }
die()  { printf '%s\n' "  ${RED}xx${OFF}  $*" >&2; exit 1; }

# ------------------------------------------------------------- interpreter
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
else
  warn ".venv not found - falling back to the system Python"
  warn "run ./install.sh first for an isolated, reproducible setup"
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] || die "no Python found. Run ./install.sh"
fi

"$PY" -c 'import numpy, pandas, sklearn, yaml, flask' 2>/dev/null \
  || die "dependencies are missing. Run ./install.sh"

CORPUS="data/corpus.npz"
DEMO="data/demo/capture.pcap"
MODEL_DIR="artifacts/model-linear"
[ -d "artifacts/model-transformer" ] && MODEL_DIR="artifacts/model-transformer"

have_torch() { "$PY" -c 'import torch' >/dev/null 2>&1; }

banner() {
  printf '\n%s\n' "${BOLD}  ATDRPS${OFF}  world-model network attack forecasting"
  printf '%s\n'   "${DIM}  SIH 2026 | PS 26153 | NTRO | Team Bell Labs${OFF}"
  if have_torch; then
    printf '%s\n' "${DIM}  backend: temporal transformer (PyTorch available)${OFF}"
  else
    printf '%s\n' "${DIM}  backend: linear dynamics (PyTorch not installed)${OFF}"
  fi
}

# ------------------------------------------------------------------ steps
step_demo() {
  say "generating a labelled synthetic capture"
  "$PY" -m atdrps.cli synth --out data/demo --seed 4242 --duration 2400 --campaigns 2 \
    || die "capture generation failed"
  ok "data/demo/capture.pcap  (ground truth in capture.timeline.json)"
}

step_corpus() {
  if [ -f "$CORPUS" ]; then
    printf '\n  %s already exists. Rebuild it? [y/N] ' "$CORPUS"
    read -r reply
    case "$reply" in [yY]*) ;; *) ok "keeping the existing corpus"; return 0 ;; esac
  fi
  say "building the training corpus  ${DIM}(48 captures, parallel across all CPU cores)${OFF}"
  "$PY" -m atdrps.cli corpus --captures 48 --out "$CORPUS" || die "corpus build failed"
  ok "$CORPUS"
}

step_train() {
  [ -f "$CORPUS" ] || { warn "no corpus yet"; step_corpus; }
  say "training the world model"
  "$PY" -m atdrps.cli train --corpus "$CORPUS" --out artifacts/model || die "training failed"
  ok "artifacts/model"
}

step_benchmark() {
  [ -f "$CORPUS" ] || { warn "no corpus yet"; step_corpus; }
  say "training every model and comparing them"
  printf '%s\n' "${DIM}  persistence, logistic regression (static + context), world model${OFF}"
  "$PY" -m atdrps.cli benchmark --corpus "$CORPUS" --split group || die "benchmark failed"
  [ -d "artifacts/model-transformer" ] && MODEL_DIR="artifacts/model-transformer"
  ok "docs/BENCHMARKS.md"
}

step_predict() {
  local capture="${1:-$DEMO}"
  [ -f "$capture" ] || { warn "no capture at $capture"; step_demo; capture="$DEMO"; }
  [ -d "$MODEL_DIR" ] || die "no trained model. Run: ./run.sh benchmark"
  say "forecasting from $capture"
  "$PY" -m atdrps.cli predict "$capture" --model "$MODEL_DIR"
}

step_dashboard() {
  [ -d "$MODEL_DIR" ] || die "no trained model. Run: ./run.sh benchmark"
  [ -f "$DEMO" ] || step_demo
  say "starting the offline dashboard"
  printf '%s\n' "  open ${BOLD}http://127.0.0.1:8501${OFF} and upload ${BOLD}$DEMO${OFF}"
  printf '%s\n' "${DIM}  nothing leaves this machine. Ctrl-C to stop.${OFF}"
  "$PY" -m atdrps.cli serve --model "$MODEL_DIR" --host 127.0.0.1 --port 8501
}

step_test() {
  say "running the test suite"
  "$PY" tests/run_tests.py || die "tests failed"
}

step_clean() {
  say "removing generated files"
  printf '  this deletes %s, artifacts/ and data/demo/. Continue? [y/N] ' "$CORPUS"
  read -r reply
  case "$reply" in
    [yY]*)
      rm -rf artifacts/* data/demo "$CORPUS"
      mkdir -p artifacts && touch artifacts/.gitkeep
      ok "cleaned"
      ;;
    *) ok "nothing removed" ;;
  esac
}

step_all() {
  banner
  step_demo
  step_corpus
  step_benchmark
  step_predict "$DEMO"
  printf '\n%s\n' "${GRN}${BOLD}  Pipeline complete.${OFF}"
  printf '%s\n'   "  results  ${BOLD}docs/BENCHMARKS.md${OFF}"
  printf '%s\n\n' "  demo     ${BOLD}./run.sh dashboard${OFF}"
}

menu() {
  banner
  while true; do
    printf '\n'
    printf '   %s  Full pipeline        %s\n' "${BOLD}1${OFF}" "${DIM}capture, corpus, benchmark, forecast${OFF}"
    printf '   %s  Generate a capture   %s\n' "${BOLD}2${OFF}" "${DIM}labelled synthetic traffic with a kill chain${OFF}"
    printf '   %s  Build the corpus     %s\n' "${BOLD}3${OFF}" "${DIM}48 captures, ~9 min${OFF}"
    printf '   %s  Train the model      %s\n' "${BOLD}4${OFF}" "${DIM}world model only${OFF}"
    printf '   %s  Run the benchmark    %s\n' "${BOLD}5${OFF}" "${DIM}all models + docs/BENCHMARKS.md${OFF}"
    printf '   %s  Forecast a capture   %s\n' "${BOLD}6${OFF}" "${DIM}timeline, stage, drivers${OFF}"
    printf '   %s  Offline dashboard    %s\n' "${BOLD}7${OFF}" "${DIM}http://127.0.0.1:8501${OFF}"
    printf '   %s  Run the tests        %s\n' "${BOLD}8${OFF}" "${DIM}273 tests${OFF}"
    printf '   %s  Clean generated files\n'   "${BOLD}9${OFF}"
    printf '   %s  Quit\n'                    "${BOLD}0${OFF}"
    printf '\n  choose: '
    read -r choice
    case "$choice" in
      1) step_all ;;
      2) step_demo ;;
      3) step_corpus ;;
      4) step_train ;;
      5) step_benchmark ;;
      6) step_predict ;;
      7) step_dashboard ;;
      8) step_test ;;
      9) step_clean ;;
      0|q|Q) printf '\n'; exit 0 ;;
      *) warn "pick a number from the list" ;;
    esac
  done
}

case "${1:-menu}" in
  menu)      menu ;;
  all)       step_all ;;
  demo)      banner; step_demo ;;
  corpus)    banner; step_corpus ;;
  train)     banner; step_train ;;
  benchmark) banner; step_benchmark ;;
  predict)   banner; step_predict "${2:-$DEMO}" ;;
  dashboard|serve) banner; step_dashboard ;;
  test)      banner; step_test ;;
  clean)     banner; step_clean ;;
  -h|--help|help) usage ;;
  *) die "unknown command: $1  (try ./run.sh --help)" ;;
esac
