"""ATDRPS command line.

    atdrps synth      --out data/demo            generate a labelled capture
    atdrps corpus     --captures 48              build a training corpus
    atdrps train      --corpus data/corpus.npz   train the world model
    atdrps benchmark  --corpus data/corpus.npz   train everything and compare
    atdrps predict    capture.pcap               forecast from a capture
    atdrps serve                                 offline dashboard

Every subcommand runs entirely locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import Config, set_global_seed

__all__ = ["main", "build_parser"]

BANNER = r"""
  ATDRPS  Advanced Threat Detection and Reinforcement Protocol System
          world-model network attack forecasting | SIH 2026 PS 26153
"""


# ------------------------------------------------------------------ helpers
def _corpus_path(args) -> Path:
    return Path(args.corpus)


def _load_corpus(path: Path):
    from .data.windows import WindowedStates

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


# ---------------------------------------------------------------- commands
def cmd_synth(args) -> int:
    from .data.synth import generate_capture, write_scenario

    capture = generate_capture(seed=args.seed, duration_s=args.duration,
                               n_campaigns=args.campaigns,
                               background_intensity=args.intensity)
    info = write_scenario(args.out, capture, name=args.name)
    print(f"wrote {info['packets']} packets -> {info['pcap']}")
    print(f"ground-truth timeline -> {info['timeline']}")
    for interval in capture.timeline:
        print(f"  campaign {interval.campaign}  {interval.stage:<18} "
              f"{interval.end - interval.start:7.1f}s  {interval.note}")
    return 0


def cmd_corpus(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_corpus import build, save  # noqa: E402

    captures = build(args.captures, args.duration, args.campaigns, args.window,
                     args.intensity, args.seed, args.history)
    save(Path(args.out), captures)
    total = sum(len(c) for c in captures)
    print(f"{total} windows across {len(captures)} captures -> {args.out}")
    return 0


def cmd_train(args) -> int:
    from .models.numpy_dynamics import NumpyDynamicsWorldModel
    from .models.transformer import TemporalTransformerWorldModel, torch_available
    from .train.dataset import build_sequences, group_split, time_split

    cfg = Config.load(args.config, args.set)
    set_global_seed(cfg.seed)

    captures = _load_corpus(_corpus_path(args))
    dataset = build_sequences(captures, context=cfg.model.context,
                              horizon=cfg.model.horizon)
    splitter = group_split if args.split == "group" else time_split
    train, val, _test = splitter(dataset, cfg.train.val_frac, cfg.train.test_frac)
    print(f"train={len(train)} val={len(val)}  split={args.split}")

    backend = args.backend or cfg.model.backend
    if backend == "torch" and not torch_available():
        print("PyTorch is not installed; falling back to the numpy backend "
              "(see requirements.txt to install it)")
        backend = "numpy"

    if backend == "torch":
        model = TemporalTransformerWorldModel(
            dataset.feature_names, cfg.model.context, cfg.model.horizon,
            d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers, ff_mult=cfg.model.ff_mult,
            dropout=cfg.model.dropout,
        )
        model.fit(train, val_dataset=val, epochs=cfg.train.epochs,
                  batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                  weight_decay=cfg.train.weight_decay, grad_clip=cfg.train.grad_clip,
                  loss_weights=dict(cfg.train.loss_weights),
                  patience=cfg.train.early_stopping_patience, verbose=True)
    else:
        model = NumpyDynamicsWorldModel(dataset.feature_names, cfg.model.context,
                                        cfg.model.horizon,
                                        ridge_alpha=cfg.model.ridge_alpha)
        model.fit(train, val_dataset=val, verbose=True)

    out = Path(args.out)
    model.save(out)
    cfg.dump(out / "resolved-config.yaml")
    np.save(out / "background.npy", train.context[:256])
    print(f"model saved to {out}")
    return 0


def cmd_benchmark(args) -> int:
    from .train.train import run_benchmark, write_report

    cfg = Config.load(args.config, args.set)
    set_global_seed(cfg.seed)
    captures = _load_corpus(_corpus_path(args))
    out = run_benchmark(
        captures, context=cfg.model.context, horizon=cfg.model.horizon,
        val_frac=cfg.train.val_frac, test_frac=cfg.train.test_frac,
        split=args.split, out_dir=args.out, epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size, lr=cfg.train.lr,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers, verbose=True,
    )
    train = out["splits"][0]
    np.save(Path(args.out) / "background.npy", train.context[:256])
    print()
    print(write_report(out["payload"], args.report))
    print(f"\nreport written to {args.report}")
    return 0


def cmd_predict(args) -> int:
    from .engine.inference import ThreatForecastEngine

    cfg = Config.load(args.config, args.set)
    model_dir = Path(args.model)
    background = None
    bg_path = model_dir / "background.npy"
    if not bg_path.exists():
        bg_path = model_dir.parent / "background.npy"
    if bg_path.exists():
        background = np.load(bg_path)

    engine = ThreatForecastEngine.load(
        model_dir, background=background, window_size_s=cfg.window.size_s,
        threshold=args.threshold, top_k=cfg.explain.top_k,
        shap_samples=cfg.explain.shap_samples,
    )
    result = engine.analyse(args.capture, horizon=args.horizon,
                            explain=not args.no_explain)

    print(BANNER)
    print(f"source      : {result.source}")
    print(f"ingested    : {result.source_kind}")
    print(f"windows     : {result.n_windows}    flows: {result.n_flows}")
    for note in result.notes:
        print(f"note        : {note}")
    if not result.timeline:
        return 1
    print(f"\n{result.headline()}\n")

    print("risk timeline (one-step-ahead)")
    print(f"  {'window':>6}  {'prob':>6}  {'stage':<20} alert")
    for row in result.timeline:
        flag = "  <-- ALERT" if row["alert"] else ""
        print(f"  {row['window']:>6}  {row['infiltration_probability']:>6.3f}  "
              f"{row['stage']:<20}{flag}")

    if result.forecast:
        print(f"\nK-step forward simulation (K={result.forecast.horizon})")
        for row in result.forecast.timeline():
            print(f"  +{row['step']}  p(infiltration)={row['infiltration_probability']:.3f}  "
                  f"stage={row['stage']} ({row['stage_confidence']:.2f})")

    if result.explanation:
        print("\nwhy")
        print(result.explanation["infiltration_explanation"].to_text())
        print(f"\npredicted stage: {result.explanation['stage_description']}")

    if result.flagged_flows:
        print(f"\nflows in the peak window (top {len(result.flagged_flows)})")
        print(f"  {'source':<24} {'destination':<24} {'pkts':>6} {'bytes':>10}  flags")
        for flow in result.flagged_flows:
            print(f"  {flow['src']:<24} {flow['dst']:<24} {flow['packets']:>6} "
                  f"{flow['bytes']:>10}  {flow['flags']}")

    if args.json:
        payload = {
            "source": result.source, "headline": result.headline(),
            "timeline": result.timeline,
            "forecast": result.forecast.timeline() if result.forecast else [],
            "flagged_flows": result.flagged_flows,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson written to {args.json}")
    return 0


def cmd_serve(args) -> int:
    from .config import Config as _Config
    cfg = _Config.load(args.config, args.set)
    from app.server import create_app

    app = create_app(model_dir=args.model, config=cfg)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(BANNER)
    print(f"offline dashboard on http://{host}:{port}  (no cloud calls)")
    app.run(host=host, port=port, debug=False)
    return 0


# ----------------------------------------------------------------- parsing
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atdrps", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set window.size_s=60")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate a labelled synthetic capture")
    p.add_argument("--out", default="data/demo")
    p.add_argument("--name", default="capture")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--duration", type=float, default=3600.0)
    p.add_argument("--campaigns", type=int, default=3)
    p.add_argument("--intensity", type=float, default=0.2)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("corpus", help="build a windowed training corpus")
    p.add_argument("--captures", type=int, default=48)
    p.add_argument("--duration", type=float, default=5400.0)
    p.add_argument("--campaigns", type=int, default=4)
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--intensity", type=float, default=0.2)
    p.add_argument("--history", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="data/corpus.npz")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("train", help="train a world model")
    p.add_argument("--corpus", default="data/corpus.npz")
    p.add_argument("--backend", choices=["torch", "numpy"], default=None)
    p.add_argument("--split", choices=["group", "time"], default="group")
    p.add_argument("--out", default="artifacts/model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("benchmark", help="train every model and compare them")
    p.add_argument("--corpus", default="data/corpus.npz")
    p.add_argument("--split", choices=["group", "time"], default="group")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--report", default="docs/BENCHMARKS.md")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("predict", help="forecast from a pcap or flow CSV")
    p.add_argument("capture")
    p.add_argument("--model", default="artifacts/model-linear")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-explain", action="store_true")
    p.add_argument("--json", default=None, help="also write the result as JSON")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("serve", help="run the offline dashboard")
    p.add_argument("--model", default="artifacts/model-linear")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
