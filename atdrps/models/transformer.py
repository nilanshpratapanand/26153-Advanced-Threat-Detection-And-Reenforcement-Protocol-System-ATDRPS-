"""The temporal transformer world model -- the primary SIH deliverable.

Learns ``P(S_{t+1} | S_{t-L+1..t})`` over sequences of network states, with
three heads sharing one encoder:

``next_state``      the dynamics target -- how the network state will *change*
                    (a residual; see below)
``stage``           the MITRE ATT&CK stage of the *next* window
``infiltration``    probability that infiltration is underway next window

Training all three together is the point.  The stage head alone would be a
classifier with extra steps; forcing the same representation to also reconstruct
the next state is what makes the encoder learn transition dynamics rather than
a static signature lookup.

On attention and leakage
------------------------
The encoder attends across all ``L`` context positions without a causal mask,
and that is deliberate and safe: every position in the context is *already in
the past* relative to the target at ``t+1``.  A causal mask inside the context
would throw away information the model legitimately has at inference time. The
leakage that matters is between the context and the target, and that is
prevented by construction -- the target window is never fed to the encoder.

Residual dynamics
-----------------
The state head predicts ``S_{t+1} - S_t`` rather than ``S_{t+1}``. Any
regularisation then pulls the model toward "nothing changes", which is the
right prior for network state and is the persistence baseline; predicting the
absolute state instead pulls it toward the *mean* state, and a roll-out that
feeds mean states back into its own context oscillates within a few steps.

The blocks are written out rather than assembled from
``nn.TransformerEncoderLayer`` for one reason: the problem statement makes
explainability mandatory, and the stock layer discards its attention weights.
Here every block keeps its head-averaged attention map, which
:mod:`atdrps.explain.attention` turns into "which of the last sixteen windows
drove this forecast".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..data.schema import STAGES
from .base import WorldModel

__all__ = ["TemporalTransformerWorldModel", "torch_available", "require_torch"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The temporal transformer backend needs PyTorch.\n"
            "  CPU:      pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  CUDA 12:  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "Or run with model.backend=numpy, which needs no extra packages."
        ) from exc


def _build_modules():
    """Define the nn.Modules lazily, so importing this file never needs torch."""
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiHeadSelfAttention(nn.Module):
        """Standard scaled dot-product attention that *keeps* its weights."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            if d_model % n_heads:
                raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.last_attention: "torch.Tensor | None" = None

        def forward(self, x):                      # x: (B, L, D)
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each (B, H, L, d_head)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            weights = torch.softmax(scores, dim=-1)
            # detached copy for explainability; never participates in the graph
            self.last_attention = weights.detach().mean(dim=1)   # (B, L, L)
            out = self.dropout(weights) @ v                      # (B, H, L, d_head)
            out = out.transpose(1, 2).reshape(B, L, D)
            return self.out(out)

    class EncoderBlock(nn.Module):
        """Pre-norm residual block -- pre-norm trains stably at this depth
        without a warmup schedule to babysit."""

        def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, ff_mult * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * d_model, d_model),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x

    class WorldModelNet(nn.Module):
        def __init__(self, n_features: int, n_stages: int, context: int,
                     d_model: int, n_heads: int, n_layers: int,
                     ff_mult: int, dropout: float):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, context, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList(
                EncoderBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
            )
            self.norm = nn.LayerNorm(d_model)
            self.head_state = nn.Linear(d_model, n_features)
            self.head_stage = nn.Linear(d_model, n_stages)
            self.head_infil = nn.Linear(d_model, 1)
            self.context = context

        def forward(self, x):                      # x: (B, L, F) standardised
            if x.shape[1] != self.context:
                raise ValueError(
                    f"expected context length {self.context}, got {x.shape[1]}"
                )
            h = self.input_proj(x) + self.pos
            for block in self.blocks:
                h = block(h)
            h = self.norm(h)
            last = h[:, -1, :]                     # the most recent window's state
            return self.head_state(last), self.head_stage(last), self.head_infil(last).squeeze(-1)

        def attention_maps(self):
            return [b.attn.last_attention for b in self.blocks if b.attn.last_attention is not None]

    return torch, nn, F, WorldModelNet


class TemporalTransformerWorldModel(WorldModel):
    name = "temporal-transformer"

    def __init__(self, feature_names, context=16, horizon=5, stage_names=STAGES,
                 d_model=128, n_heads=4, n_layers=3, ff_mult=4, dropout=0.1,
                 device: str | None = None) -> None:
        super().__init__(feature_names, context, horizon, stage_names)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_mult = int(ff_mult)
        self.dropout = float(dropout)
        self.device_name = device
        self.net = None
        self.history: list[dict] = []

    # ---------------------------------------------------------------- setup
    def _device(self):
        torch = require_torch()
        if self.device_name:
            return torch.device(self.device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self):
        torch, nn, F, WorldModelNet = _build_modules()
        net = WorldModelNet(
            n_features=self.n_features, n_stages=self.n_stages, context=self.context,
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            ff_mult=self.ff_mult, dropout=self.dropout,
        ).to(self._device())
        return torch, nn, F, net

    # ------------------------------------------------------------- training
    def fit(self, dataset, val_dataset=None, epochs: int = 40, batch_size: int = 64,
            lr: float = 3e-4, weight_decay: float = 1e-4, grad_clip: float = 1.0,
            loss_weights: dict | None = None, class_weight: str | None = "balanced",
            patience: int = 8, verbose: bool = True, **kwargs):
        from ..train.dataset import class_weights as _class_weights

        if len(dataset) == 0:
            raise ValueError("cannot fit on an empty dataset")
        torch, nn, F, net = self._build()
        self.net = net
        device = self._device()
        weights = {"next_state": 1.0, "stage": 1.0, "infiltration": 1.0}
        weights.update(loss_weights or {})

        self.standardiser.fit(dataset.context)
        Xc = torch.tensor(self.standardiser.transform(dataset.context), device=device)
        # residual dynamics target
        Ys = torch.tensor(
            self.standardiser.transform(dataset.target_state)
            - self.standardiser.transform(dataset.context[:, -1, :]),
            device=device,
        )
        Yst = torch.tensor(dataset.target_stage[:, 0], device=device, dtype=torch.long)
        Yin = torch.tensor(dataset.target_infil[:, 0], device=device, dtype=torch.float32)
        M = torch.tensor(dataset.target_mask[:, 0], device=device, dtype=torch.bool)

        if class_weight == "balanced":
            cw = _class_weights(dataset.target_stage[:, 0], self.n_stages,
                                dataset.target_mask[:, 0])
            stage_w = torch.tensor(cw, device=device, dtype=torch.float32)
            positives = float(dataset.target_infil[:, 0][dataset.target_mask[:, 0]].mean() or 0.5)
            pos_weight = torch.tensor(
                [(1 - positives) / max(positives, 1e-6)], device=device, dtype=torch.float32
            )
        else:
            stage_w, pos_weight = None, None

        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        n = len(dataset)
        best, best_state, bad_epochs = float("inf"), None, 0
        self.history = []

        for epoch in range(epochs):
            net.train()
            order = torch.randperm(n, device=device)
            totals = {"loss": 0.0, "state": 0.0, "stage": 0.0, "infil": 0.0}
            batches = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                pred_state, logit_stage, logit_infil = net(Xc[idx])

                loss_state = F.mse_loss(pred_state, Ys[idx])
                m = M[idx]
                if m.any():
                    loss_stage = F.cross_entropy(logit_stage[m], Yst[idx][m], weight=stage_w)
                    loss_infil = F.binary_cross_entropy_with_logits(
                        logit_infil[m], Yin[idx][m], pos_weight=pos_weight
                    )
                else:
                    loss_stage = pred_state.sum() * 0.0
                    loss_infil = pred_state.sum() * 0.0

                loss = (weights["next_state"] * loss_state
                        + weights["stage"] * loss_stage
                        + weights["infiltration"] * loss_infil)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()

                totals["loss"] += float(loss.item())
                totals["state"] += float(loss_state.item())
                totals["stage"] += float(loss_stage.item())
                totals["infil"] += float(loss_infil.item())
                batches += 1
            sched.step()

            record = {k: v / max(batches, 1) for k, v in totals.items()}
            record["epoch"] = epoch + 1
            monitor = record["loss"]
            if val_dataset is not None and len(val_dataset):
                record["val_loss"] = self._val_loss(val_dataset, weights, stage_w, pos_weight)
                monitor = record["val_loss"]
            self.history.append(record)
            if verbose:
                extra = f" val={record['val_loss']:.4f}" if "val_loss" in record else ""
                print(f"  epoch {epoch + 1:>3}/{epochs} loss={record['loss']:.4f} "
                      f"(state={record['state']:.4f} stage={record['stage']:.4f} "
                      f"infil={record['infil']:.4f}){extra}", flush=True)

            if monitor < best - 1e-5:
                best, bad_epochs = monitor, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch + 1} "
                              f"(no improvement for {patience} epochs)", flush=True)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.fitted = True
        return self

    def _val_loss(self, val_dataset, weights, stage_w, pos_weight) -> float:
        torch, nn, F, _ = _build_modules()
        device = self._device()
        self.net.eval()
        with torch.no_grad():
            Xc = torch.tensor(self.standardiser.transform(val_dataset.context), device=device)
            Ys = torch.tensor(
                self.standardiser.transform(val_dataset.target_state)
                - self.standardiser.transform(val_dataset.context[:, -1, :]),
                device=device,
            )
            Yst = torch.tensor(val_dataset.target_stage[:, 0], device=device, dtype=torch.long)
            Yin = torch.tensor(val_dataset.target_infil[:, 0], device=device, dtype=torch.float32)
            M = torch.tensor(val_dataset.target_mask[:, 0], device=device, dtype=torch.bool)
            pred_state, logit_stage, logit_infil = self.net(Xc)
            loss = weights["next_state"] * F.mse_loss(pred_state, Ys)
            if M.any():
                loss = loss + weights["stage"] * F.cross_entropy(
                    logit_stage[M], Yst[M], weight=stage_w)
                loss = loss + weights["infiltration"] * F.binary_cross_entropy_with_logits(
                    logit_infil[M], Yin[M], pos_weight=pos_weight)
            return float(loss.item())

    # ------------------------------------------------------------ inference
    def _forward(self, context: np.ndarray):
        torch, _, _, _ = _build_modules()
        if self.net is None:
            raise RuntimeError("model is not fitted")
        arr = np.asarray(context, dtype=np.float32)
        single = arr.ndim == 2
        if single:
            arr = arr[None, ...]
        arr = arr[:, -self.context:, :]
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.standardiser.transform(arr), device=self._device())
            delta, stage_logits, infil_logit = self.net(x)
            probs = torch.softmax(stage_logits, dim=-1).cpu().numpy()
            infil = torch.sigmoid(infil_logit).cpu().numpy()
            # the head predicts a change; add it back to the last observed state
            last = x[:, -1, :].cpu().numpy()
            state = self.standardiser.inverse(last + delta.cpu().numpy())
        return (state[0], probs[0], float(infil[0])) if single else (state, probs, infil)

    def predict_next(self, context: np.ndarray) -> np.ndarray:
        return self._forward(context)[0].astype(np.float32)

    def predict_next_batch(self, context: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(context))[0]

    def heads(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        _, probs, infil = self._forward(context)
        return probs.astype(np.float32), float(infil)

    def heads_batch(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, probs, infil = self._forward(np.asarray(context))
        return probs.astype(np.float32), np.asarray(infil, dtype=np.float32)

    def attention(self, context: np.ndarray) -> np.ndarray:
        """Head-averaged attention from the last block, for the most recent
        window's query: ``(L,)``, summing to 1 across the context.

        Reads as "how much of this forecast came from each of the last L
        windows" -- which is the question an analyst actually asks.
        """
        self._forward(context)
        maps = self.net.attention_maps()
        if not maps:
            return np.zeros(self.context, dtype=np.float32)
        last = maps[-1].cpu().numpy()          # (B, L, L)
        return last[0, -1, :].astype(np.float32)

    def attention_all_layers(self, context: np.ndarray) -> np.ndarray:
        """``(n_layers, L)`` -- the same query row from every block."""
        self._forward(context)
        maps = self.net.attention_maps()
        if not maps:
            return np.zeros((self.n_layers, self.context), dtype=np.float32)
        return np.stack([m.cpu().numpy()[0, -1, :] for m in maps]).astype(np.float32)

    # --------------------------------------------------------- persistence
    def config_dict(self) -> dict:
        base = super().config_dict()
        base.update({
            "d_model": self.d_model, "n_heads": self.n_heads,
            "n_layers": self.n_layers, "ff_mult": self.ff_mult,
            "dropout": self.dropout, "history": self.history,
        })
        return base

    def save(self, path: str | Path) -> None:
        torch = require_torch()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._write_config(path / "config.json")
        if self.net is not None:
            torch.save(self.net.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalTransformerWorldModel":
        torch = require_torch()
        path = Path(path)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        model = cls(
            feature_names=config["feature_names"], context=config["context"],
            horizon=config["horizon"], stage_names=tuple(config["stage_names"]),
            d_model=config["d_model"], n_heads=config["n_heads"],
            n_layers=config["n_layers"], ff_mult=config["ff_mult"],
            dropout=config["dropout"],
        )
        model.history = config.get("history", [])
        if config.get("standardiser"):
            from .base import Standardiser
            model.standardiser = Standardiser.from_state_dict(config["standardiser"])
        _, _, _, net = model._build()
        net.load_state_dict(torch.load(path / "weights.pt", map_location=model._device()))
        model.net = net
        model.fitted = True
        return model
