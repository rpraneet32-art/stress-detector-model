"""
models/bilstm.py — Bidirectional LSTM architecture (Section 6.2).

Exact spec:
  Input transposed to (B, 384, 3)
  Bi-LSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)
  Concatenate final forward + backward hidden states → (B, 128)
  Linear(128 → 64) → ReLU → Linear(64 → 2)

Input:  (B, 3, 384)   — channels-first from DataLoader
Output: (B, 2)        — raw logits
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """Bidirectional LSTM for stress detection."""

    def __init__(
        self,
        input_size:  int = 3,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        n_classes:   int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        # Bidirectional → 2 × hidden_size per direction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 384)  — channels-first

        Returns
        -------
        logits : (B, 2)
        """
        # Spec: input transposed to (B, T, C)
        x = x.permute(0, 2, 1)                    # (B, 384, 3)

        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden_size) — forward & backward interleaved
        # Take the last layer's forward and backward final hidden states
        # Layer index for last layer: forward = -2, backward = -1
        h_fwd = h_n[-2]                            # (B, 64)
        h_bwd = h_n[-1]                            # (B, 64)
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 128)

        return self.classifier(h_cat)              # (B, 2)


# ── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import WINDOW_SAMPLES

    model = BiLSTM()
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    dummy = torch.randn(8, 3, WINDOW_SAMPLES)
    out   = model(dummy)
    assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"
    print(f"Forward pass OK: input {tuple(dummy.shape)} → output {tuple(out.shape)}")
