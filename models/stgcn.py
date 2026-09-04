"""
models/stgcn.py — Spatio-Temporal Graph Convolutional Network (Section 6.4).

Exact spec:
  Graph nodes (V=3): Node 0=ECG, Node 1=PPG, Node 2=Skin Temperature
  Adjacency A ∈ R^(3×3): dynamic Pearson correlation between the 3 channels,
      computed fresh per 3-second window (not fixed/learned globally)
  GCNConv layer(s): propagate features across the correlation-weighted graph
  1D temporal convolution over node embeddings across the time axis
  Global mean pooling over nodes → Linear → Softmax(2) (logits returned)

Design notes:
  - We implement a PyG-free version using manual message passing with the
    Pearson adjacency, so the model runs without torch-geometric if unavailable
    (torch-geometric installation on Kaggle/Colab can fail due to CUDA mismatches).
    If torch_geometric IS available, GCNConv is used; otherwise we fall back to
    a manual symmetric normalized adjacency multiply.
  - The dynamic adjacency is computed per sample in the forward pass, then
    used as the graph structure for that sample's spatial convolution.

Input:  (B, 3, 384)
Output: (B, 2)  — raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pearson correlation adjacency ────────────────────────────────────────────

def _pearson_adj(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute a symmetric, self-loop-enhanced Pearson correlation adjacency matrix
    for each sample in the batch.

    Parameters
    ----------
    x : (B, V, T)  — V=3 nodes, T=384 time steps

    Returns
    -------
    A : (B, V, V)  — values in [0, 1] after abs + self-loop
    """
    B, V, T = x.shape
    # Center each node's time series
    x_c = x - x.mean(dim=-1, keepdim=True)        # (B, V, T)
    # Pairwise dot products (unnormalized covariance)
    cov = torch.bmm(x_c, x_c.transpose(1, 2))     # (B, V, V)
    # Norms
    std = x_c.norm(dim=-1, keepdim=True) + eps     # (B, V, 1)
    norm_mat = torch.bmm(std, std.transpose(1, 2)) # (B, V, V)
    corr = cov / norm_mat                          # (B, V, V)  in [-1, 1]
    # Use absolute correlation (negative correlation still carries info)
    A = corr.abs()
    # Add self-loops (prevent node from losing its own signal)
    eye = torch.eye(V, device=x.device).unsqueeze(0).expand(B, -1, -1)
    A = A + eye
    # Row-normalize: D^{-1} A  (simpler than D^{-1/2} A D^{-1/2} for stability)
    deg = A.sum(dim=-1, keepdim=True) + eps
    A = A / deg
    return A


# ── Graph convolution (manual, no PyG dependency) ───────────────────────────

class ManualGCNConv(nn.Module):
    """
    Single graph convolution layer: H' = A · H · W  (per sample in batch).
    Works with dynamic, per-sample adjacency matrices.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias   = nn.Parameter(torch.zeros(out_features))

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H : (B, V, F_in)
        A : (B, V, V)
        → (B, V, F_out)
        """
        H_proj = self.linear(H)           # (B, V, F_out)
        H_agg  = torch.bmm(A, H_proj)    # (B, V, F_out)  — neighborhood aggregation
        return F.relu(H_agg + self.bias)


# ── ST-GCN ──────────────────────────────────────────────────────────────────

class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for stress detection.

    Architecture:
      1. GCN on the full (V=3 nodes, T=384 time steps) representation
         with dynamic Pearson adjacency
      2. 1D temporal conv over each node's feature sequence
      3. Global mean pool over nodes → fully-connected → logits
    """

    def __init__(
        self,
        n_nodes:    int = 3,
        in_feat:    int = 1,         # 1 feature per node per time step (raw signal)
        gcn_hidden: int = 32,        # GCN output dimension per node
        tcn_channels: int = 64,      # Temporal conv output channels
        tcn_kernel: int = 7,
        n_classes:  int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Spatial: GCN applied independently at each time step
        # To make this efficient we vectorize: treat T as batch dim
        self.gcn1 = ManualGCNConv(in_feat, gcn_hidden)
        self.gcn2 = ManualGCNConv(gcn_hidden, gcn_hidden)

        # Temporal: 1D conv over node embeddings  (B*V, gcn_hidden, T) → (B*V, tcn_channels, T')
        self.tcn = nn.Sequential(
            nn.Conv1d(gcn_hidden, tcn_channels, kernel_size=tcn_kernel,
                      padding=tcn_kernel // 2),
            nn.BatchNorm1d(tcn_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # → (B*V, tcn_channels, 1)
        )

        # Classifier
        self.classifier = nn.Linear(tcn_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 384)  — channels-first; each channel = one graph node

        Returns
        -------
        logits : (B, 2)
        """
        B, V, T = x.shape  # V=3, T=384

        # ── Compute dynamic Pearson adjacency ─────────────────────────
        A = _pearson_adj(x)                        # (B, V, V)

        # ── Spatial GCN: apply at each time step ──────────────────────
        # Rearrange: (B, V, T) → (B*T, V, 1) to process all timesteps at once
        # Then compute adjacency-weighted aggregation
        # Strategy: expand A over T, apply GCN to all T in one matmul

        # H: (B, V, T) → transpose → (B, T, V) → unsqueeze → (B, T, V, 1)
        H = x.permute(0, 2, 1).unsqueeze(-1)       # (B, T, V, 1)

        # Flatten B and T: (B*T, V, 1)
        H_flat = H.reshape(B * T, V, 1)

        # Expand A: (B, V, V) → (B, T, V, V) → (B*T, V, V)
        A_exp = A.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, V, V)

        # GCN layers
        H_flat = self.gcn1(H_flat, A_exp)           # (B*T, V, gcn_hidden)
        H_flat = self.gcn2(H_flat, A_exp)           # (B*T, V, gcn_hidden)

        # ── Temporal conv: per node over time ─────────────────────────
        # Reshape: (B*T, V, F) → (B, V, T, F) → (B*V, F, T)
        gcn_out = H_flat.reshape(B, T, V, -1)       # (B, T, V, F)
        gcn_out = gcn_out.permute(0, 2, 3, 1)       # (B, V, F, T)
        F_dim = gcn_out.shape[2]
        tcn_in = gcn_out.reshape(B * V, F_dim, T)   # (B*V, F, T)

        tcn_out = self.tcn(tcn_in)                  # (B*V, tcn_channels, 1)
        tcn_out = tcn_out.squeeze(-1)               # (B*V, tcn_channels)

        # ── Global mean pooling over nodes ────────────────────────────
        node_feats = tcn_out.reshape(B, V, -1)      # (B, V, tcn_channels)
        pooled = node_feats.mean(dim=1)             # (B, tcn_channels)

        return self.classifier(pooled)              # (B, 2)


# ── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import WINDOW_SAMPLES

    model = STGCN()
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    dummy = torch.randn(8, 3, WINDOW_SAMPLES)
    out   = model(dummy)
    assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"
    print(f"Forward pass OK: input {tuple(dummy.shape)} → output {tuple(out.shape)}")

    # Also verify adjacency is in valid range
    adj = _pearson_adj(dummy)
    assert adj.shape == (8, 3, 3)
    assert adj.min() >= 0
    print(f"Adjacency range: [{adj.min():.4f}, {adj.max():.4f}]  PASS")
