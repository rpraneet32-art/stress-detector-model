"""
models/cnn.py — 1D-CNN architecture (Section 6.1).

Exact spec:
  Conv1D(in=3,  out=32,  k=7, pad=3) → BatchNorm1d → ReLU → MaxPool1d(2)
  Conv1D(32,    out=64,  k=5, pad=2) → BatchNorm1d → ReLU → MaxPool1d(2)
  Conv1D(64,    out=128, k=3, pad=1) → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)
  Linear(128 → 32) → Dropout(0.3) → Linear(32 → 2)

Input:  (B, 3, 384)
Output: (B, 2)  — raw logits
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for stress detection."""

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=3,  out_channels=32,  kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64,  kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),   # → (B, 128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                           # → (B, 128)
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 384)

        Returns
        -------
        logits : (B, 2)
        """
        feat = self.features(x)    # (B, 128, 1)
        return self.classifier(feat)


# ── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import WINDOW_SAMPLES

    model = CNN1D()
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    dummy = torch.randn(8, 3, WINDOW_SAMPLES)
    out   = model(dummy)
    assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"
    print(f"Forward pass OK: input {tuple(dummy.shape)} → output {tuple(out.shape)}")
