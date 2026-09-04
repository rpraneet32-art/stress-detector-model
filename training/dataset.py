"""
training/dataset.py — PyTorch Dataset and LOSO split utilities.

Provides:
  - StressDataset: wraps (N, 3, 384) tensors and (N,) labels
  - loso_split:   Leave-One-Subject-Out fold generator
  - compute_class_weights: for weighted cross-entropy (Section 8)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Iterator, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_SUBJECTS, VAL_SPLIT, BATCH_SIZE, GLOBAL_SEED


class StressDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-windowed tensors and labels.

    Parameters
    ----------
    tensors     : (N, 3, 384) float32
    labels      : (N,)        int32 / long
    subject_ids : (N,)        int32   — kept for reference / downstream filtering
    features    : (N, F)      float32 — classical feature vectors (optional)
    """

    def __init__(
        self,
        tensors:     np.ndarray,
        labels:      np.ndarray,
        subject_ids: np.ndarray,
        features:    np.ndarray = None,
    ):
        self.tensors     = torch.from_numpy(tensors).float()
        self.labels      = torch.from_numpy(labels).long()
        self.subject_ids = torch.from_numpy(subject_ids).long()
        self.features    = torch.from_numpy(features).float() if features is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "tensor":     self.tensors[idx],
            "label":      self.labels[idx],
            "subject_id": self.subject_ids[idx],
        }
        if self.features is not None:
            item["features"] = self.features[idx]
        return item


def loso_split(
    dataset_dict: dict,
    test_subject: int,
    val_split: float = VAL_SPLIT,
    seed: int = GLOBAL_SEED,
) -> Tuple[dict, dict, dict]:
    """
    Split the full dataset into train / val / test for one LOSO fold.

    - test  : all windows from test_subject
    - train : remaining 14 subjects, (1 - val_split) fraction
    - val   : remaining 14 subjects, val_split fraction (80/20 within-pool split)

    Returns three dicts with keys: tensors, features, labels, subject_ids
    """
    sids = dataset_dict["subject_ids"]

    test_mask  = sids == test_subject
    train_mask = ~test_mask

    train_sids    = sids[train_mask]
    train_tensors = dataset_dict["tensors"][train_mask]
    train_feats   = dataset_dict["features"][train_mask]
    train_labels  = dataset_dict["labels"][train_mask]

    # Reproducible shuffle then split
    rng = np.random.default_rng(seed)
    idx = np.arange(train_mask.sum())
    rng.shuffle(idx)
    n_val = int(len(idx) * val_split)
    val_idx   = idx[:n_val]
    tr_idx    = idx[n_val:]

    def _slice(key, indices):
        return dataset_dict[key][train_mask][indices]

    train_dict = {
        "tensors":     train_tensors[tr_idx],
        "features":    train_feats[tr_idx],
        "labels":      train_labels[tr_idx],
        "subject_ids": train_sids[tr_idx],
    }
    val_dict = {
        "tensors":     train_tensors[val_idx],
        "features":    train_feats[val_idx],
        "labels":      train_labels[val_idx],
        "subject_ids": train_sids[val_idx],
    }
    test_dict = {
        "tensors":     dataset_dict["tensors"][test_mask],
        "features":    dataset_dict["features"][test_mask],
        "labels":      dataset_dict["labels"][test_mask],
        "subject_ids": sids[test_mask],
    }
    return train_dict, val_dict, test_dict


def compute_class_weights(labels: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    Used for weighted cross-entropy (Section 8).
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    total  = counts.sum()
    weights = total / (n_classes * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(
    train_dict: dict,
    val_dict: dict,
    test_dict: dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from fold dicts."""

    def _dl(d, shuffle):
        ds = StressDataset(
            tensors=d["tensors"],
            labels=d["labels"],
            subject_ids=d["subject_ids"],
            features=d.get("features"),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    return _dl(train_dict, True), _dl(val_dict, False), _dl(test_dict, False)
