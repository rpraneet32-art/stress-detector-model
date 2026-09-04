"""
models/transformer.py — Temporal Transformer architecture (Section 6.3).

Exact spec:
  Linear(3 → 64)  [projects channel dim to d_model=64]
  Add learnable/sinusoidal positional encoding, shape (384, 64)
  3× TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, norm_first=True)
  Global average pool over the temporal dimension → (B, 64)
  Linear(64 → 2)

Input:  (B, 3, 384)   — channels-first
Output: (B, 2)        — raw logits

Implementation note: norm_first=True uses Pre-LN (LayerNorm before attention),
which the spec explicitly requires. PyTorch ≥ 1.11 supports this natively.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class TemporalTransformer(nn.Module):
    """Temporal Transformer Encoder for stress detection."""

    def __init__(
        self,
        in_channels:    int = 3,
        d_model:        int = 64,
        nhead:          int = 4,
        dim_feedforward: int = 128,
        num_layers:     int = 3,
        dropout:        float = 0.1,
        n_classes:      int = 2,
    ):
        super().__init__()

        # Channel projection: (B, T, 3) → (B, T, 64)
        self.input_proj = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,    # Pre-LN as required by spec
            batch_first=True,   # (B, T, d_model) convention
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 384)  — channels-first

        Returns
        -------
        logits : (B, 2)
        """
        # (B, 3, 384) → (B, 384, 3)
        x = x.permute(0, 2, 1)

        # Project channels to d_model: (B, 384, 3) → (B, 384, 64)
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_enc(x)                        # (B, 384, 64)

        # Transformer encoder
        x = self.encoder(x)                        # (B, 384, 64)

        # Global average pool over temporal dim
        x = x.mean(dim=1)                          # (B, 64)

        return self.classifier(x)                  # (B, 2)


# ── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import WINDOW_SAMPLES

    model = TemporalTransformer()
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    dummy = torch.randn(8, 3, WINDOW_SAMPLES)
    out   = model(dummy)
    assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"
    print(f"Forward pass OK: input {tuple(dummy.shape)} → output {tuple(out.shape)}")
