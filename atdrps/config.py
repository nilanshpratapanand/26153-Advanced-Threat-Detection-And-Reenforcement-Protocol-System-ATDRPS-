"""Configuration loading for ATDRPS.

A run is reproducible from ``(config file, dataset, seed)``.  Everything the
pipeline needs -- window width, model size, loss weights, MITRE stage list --
lives in a single YAML document so that nothing is hard-coded in the modules
themselves.

Typical use::

    cfg = Config.load()                       # configs/default.yaml
    cfg = Config.load("configs/fast.yaml")
    cfg = Config.load(overrides=["model.backend=numpy", "window.size_s=10"])

    cfg.window.size_s        # attribute access
    cfg["window"]["size_s"]  # mapping access
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = ["Config", "DEFAULT_CONFIG_PATH", "set_global_seed"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _coerce(text: str) -> Any:
    """Turn a CLI override string into a python scalar."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [_coerce(part.strip()) for part in text.split(",")]
    return text


def _deep_merge(base: dict, extra: Mapping) -> dict:
    """Recursively merge ``extra`` into ``base`` (returns a new dict)."""
    out = dict(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config(dict):
    """A dict that also supports dotted attribute access.

    Nested mappings are wrapped lazily so ``cfg.train.loss_weights.stage``
    works without the caller knowing how deep the tree is.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - attribute error is the contract
            raise AttributeError(
                f"no config key {name!r} (available: {sorted(self.keys())})"
            ) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        path: str | os.PathLike | None = None,
        overrides: Iterable[str] | None = None,
    ) -> "Config":
        """Read a YAML config, layering it on top of ``configs/default.yaml``.

        ``overrides`` is a list of ``dotted.key=value`` strings, applied last,
        so the CLI can tweak a single knob without a new file.
        """
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        if path is not None and Path(path) != DEFAULT_CONFIG_PATH:
            with open(path, "r", encoding="utf-8") as fh:
                data = _deep_merge(data, yaml.safe_load(fh) or {})

        cfg = cls(data)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"override must look like key=value, got {item!r}")
            dotted, _, raw = item.partition("=")
            cfg.set(dotted.strip(), _coerce(raw.strip()))
        return cfg

    # ------------------------------------------------------------ dotted get
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    # --------------------------------------------------------------- helpers
    def to_dict(self) -> dict:
        def unwrap(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj

        return unwrap(self)

    def dump(self, path: str | os.PathLike) -> None:
        """Persist the *resolved* config next to a model, for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @property
    def artifacts_dir(self) -> Path:
        d = REPO_ROOT / str(self.get_path("paths.artifacts", "artifacts"))
        d.mkdir(parents=True, exist_ok=True)
        return d


def set_global_seed(seed: int) -> None:
    """Seed every RNG we might touch.  Torch is seeded only if importable."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:  # optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
