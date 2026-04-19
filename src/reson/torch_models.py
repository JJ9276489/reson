from __future__ import annotations

from typing import Any


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError("Torch models require `python -m pip install '.[ml]'`") from exc
    return torch, nn


def _make_module_base():
    _, nn = require_torch()
    return nn.Module


class _PermuteBatchTimeFeature(_make_module_base()):
    def forward(self, x):
        return x.transpose(1, 2)


class _TinyTcn(_make_module_base()):
    def __init__(self, input_dim: int, hidden: int):
        _, nn = require_torch()
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.act(self.in_proj(x))
        h = self.act(self.conv1(h)[..., : x.shape[-1]])
        h = self.act(self.conv2(h)[..., : x.shape[-1]])
        return self.out(self.pool(h).squeeze(-1))


class _TinyTransformer(_make_module_base()):
    def __init__(self, input_dim: int, hidden: int, heads: int, layers: int, seq_len: int):
        torch, nn = require_torch()
        super().__init__()
        heads = max(1, heads)
        if hidden % heads != 0:
            hidden += heads - (hidden % heads)
        self.proj = nn.Linear(input_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(layers, 1))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.proj(x)
        h = h + self.pos[:, : h.shape[1], :]
        h = self.encoder(h)
        return self.out(h.mean(dim=1))


def build_torch_model(model_type: str, config: dict[str, Any]):
    _, nn = require_torch()
    input_dim = int(config["input_dim"])
    hidden = int(config.get("hidden", 16))
    seq_len = int(config.get("seq_len", 16))

    if model_type == "cnn":
        return nn.Sequential(
            _PermuteBatchTimeFeature(),
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    if model_type == "tcn":
        return _TinyTcn(input_dim=input_dim, hidden=hidden)

    if model_type == "transformer":
        return _TinyTransformer(
            input_dim=input_dim,
            hidden=hidden,
            heads=int(config.get("heads", 2)),
            layers=int(config.get("layers", 1)),
            seq_len=seq_len,
        )

    raise ValueError(f"unknown torch model_type={model_type!r}")
