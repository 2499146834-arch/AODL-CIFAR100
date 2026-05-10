"""Training loop with AMP, early stopping, cosine annealing, and multi-seed support."""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from config import (
    DEVICE, EARLY_STOP_PATIENCE, CHECKPOINT_DIR,
    USE_COSINE_ANNEALING, COSINE_T_MAX, COSINE_ETA_MIN, USE_AMP,
)


def set_seed(seed: int):
    """Fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster with AMP


def train_one_epoch(model, loader, optimizer, criterion, scaler=None) -> float:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):
            loss = criterion(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(enabled=USE_AMP):
            logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    acc = correct / total
    loss = total_loss / total if criterion is not None else None
    return acc, loss


def train(
    model, train_loader, test_loader,
    *, lr, epochs, loss_fn, name="model", save_best=True,
    use_cosine=USE_COSINE_ANNEALING, seed=None,
) -> dict:
    """Train with early stopping + optional cosine annealing. Returns history dict."""

    if seed is not None:
        set_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=USE_AMP)

    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN,
        )
    else:
        scheduler = None

    history = {"train_loss": [], "test_acc": [], "test_loss": []}
    best_acc = 0.0
    best_path = CHECKPOINT_DIR / f"{name}_best.pth"
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
        test_acc, test_loss = evaluate(model, test_loader, loss_fn)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            patience_counter = 0
            if save_best:
                torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_acc: {test_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"lr: {lr_now:.2e}"
              f"{'  *best*' if is_best else ''}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    history["best_acc"] = best_acc
    history["best_epoch"] = history["test_acc"].index(best_acc) + 1
    history["stopped_epoch"] = epoch
    return history
