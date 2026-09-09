"""
training/trainer.py — Deep learning training loop (Section 8).

Implements for all four DL architectures:
  - AdamW optimizer, lr=1e-3, β1=0.9, β2=0.999, wd=1e-2
  - Cosine annealing LR schedule
  - Weighted cross-entropy (class weights from class balance)
  - Early stopping on validation loss (patience = EARLY_STOP_PATIENCE)
  - Per-fold logging via run_logger
  - Checkpoint saving per fold
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LEARNING_RATE, BETAS, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, BATCH_SIZE,
    CHECKPOINT_DIR, GLOBAL_SEED, seed_everything,
)
from training.metrics import compute_metrics, benchmark_latency
from logger import run_logger


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    class_weights: torch.Tensor,
    model_name: str,
    fold: int,
    config: dict,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """
    Train one LOSO fold and return metrics dict.

    Parameters
    ----------
    model         : initialized (untrained) model
    train_loader  : DataLoader for training windows
    val_loader    : DataLoader for validation windows
    test_loader   : DataLoader for held-out test subject
    class_weights : (2,) tensor for weighted cross-entropy
    model_name    : string ID for logging
    fold          : LOSO fold index (= test subject ID)
    config        : hyperparameter dict for logging
    seed          : random seed

    Returns
    -------
    dict with train/val loss curves + all test metrics
    """
    seed_everything(seed)
    device = _get_device()
    model  = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    best_ckpt_path = None
    t_start        = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # -- Train --
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch["tensor"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        scheduler.step()

        # -- Validate --
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # -- Early stopping --
        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            patience_count = 0
            # Save best checkpoint
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            ckpt_name = f"{model_name}_fold{fold}_best.pt"
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "config":      config,
                "seed":        seed,
                "fold":        fold,
            }, best_ckpt_path)
        else:
            patience_count += 1

        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"    Early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})", flush=True)
            break

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )

    wall_time = time.time() - t_start

    # ── Load best checkpoint and evaluate on test set ────────────────────
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = _evaluate_test(model, test_loader, device)

    # ── CPU latency benchmark ────────────────────────────────────────────
    latency = benchmark_latency(model, window_shape=(1, 3, 384), device="cpu")
    print(
        f"    CPU latency: {latency['latency_mean_ms']:.1f} ms "
        f"({'PASS' if latency['latency_pass'] else 'FAIL >50ms'})",
        flush=True
    )

    # ── Log everything ───────────────────────────────────────────────────
    all_metrics = {**test_metrics, **latency}
    run_logger.log(
        model_name=model_name,
        fold=fold,
        config=config,
        seed=seed,
        metrics=all_metrics,
        checkpoint_path=best_ckpt_path,
        wall_time_sec=round(wall_time, 2),
        extra={
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_val_loss": best_val_loss,
            "stopped_epoch": len(train_losses),
        },
    )

    return {
        "fold":         fold,
        "metrics":      all_metrics,
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "wall_time_sec": wall_time,
        "checkpoint":   best_ckpt_path,
    }


def _evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["tensor"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
    return total_loss / max(len(loader.dataset), 1)


def _evaluate_test(model, loader, device) -> dict:
    model.eval()
    all_preds = []
    all_probs = []
    all_true  = []
    with torch.no_grad():
        for batch in loader:
            x = batch["tensor"].to(device)
            y = batch["label"]
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return compute_metrics(y_true, y_pred, y_prob)
