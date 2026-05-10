"""Comparative experiments with multi-seed runs (mean ± std)."""

import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import get_dataloaders
from model import CIFAR100ResNet
from train import train, set_seed
from config import (
    DEVICE, SEEDS, EXP_LR_LIST, EXP_BS_LIST, EXP_EPOCHS,
    DEFAULT_LR, DEFAULT_BS, LABEL_SMOOTHING, N_RUNS,
    RESULTS_DIR, FIGURES_DIR, NUM_WORKERS, USE_COSINE_ANNEALING,
)


def _new_model():
    return CIFAR100ResNet().to(DEVICE)


def _multi_seed_run(train_fn, seeds=None):
    """Run train_fn(seed) once per seed, return list of history dicts."""
    if seeds is None:
        seeds = SEEDS[:N_RUNS]
    histories = []
    for seed in seeds:
        set_seed(seed)
        hist = train_fn(seed)
        histories.append(hist)
    return histories


def _mean_curve(curves):
    """curves: list of lists (runs × epochs). Pad to shortest, return mean & std."""
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return arr.mean(axis=0), arr.std(axis=0)


# ═══════════════════════════════════════════════════════════════════
# Experiment 1: Loss Function Comparison (multi-seed)
# ═══════════════════════════════════════════════════════════════════
def exp_loss_function():
    print("\n" + "=" * 60)
    print("Experiment 1: Loss Function Comparison (multi-seed)")
    print(f"Cosine LR: {USE_COSINE_ANNEALING} | Seeds: {SEEDS[:N_RUNS]}")
    print("=" * 60)

    results = {}

    for loss_name, criterion in [
        ("CELoss", nn.CrossEntropyLoss()),
        ("LabelSmoothing_0.1", nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)),
    ]:
        print(f"\n--- {loss_name} ---")
        accs, times = [], []

        def _run(seed):
            train_loader, test_loader = get_dataloaders(DEFAULT_BS, NUM_WORKERS, seed=seed)
            model = _new_model()
            t0 = time.time()
            hist = train(model, train_loader, test_loader,
                         lr=DEFAULT_LR, epochs=50, loss_fn=criterion,
                         name=f"exp_loss_{loss_name}_s{seed}", save_best=(seed == SEEDS[0]),
                         use_cosine=USE_COSINE_ANNEALING, seed=seed)
            elapsed = time.time() - t0
            accs.append(hist["best_acc"])
            times.append(elapsed)
            print(f"  [seed={seed}] best_acc: {hist['best_acc']:.4f} @ {hist['best_epoch']}ep | "
                  f"stop: {hist['stopped_epoch']}ep | {elapsed:.0f}s")
            return hist

        histories = _multi_seed_run(_run)
        results[loss_name] = {
            "histories": histories,
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_time": np.mean(times),
        }
        print(f"  => Mean ± Std: {results[loss_name]['mean_acc']:.4f} ± {results[loss_name]['std_acc']:.4f}")

    # Plot with error bands
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, r in results.items():
        curves_acc = [h["test_acc"] for h in r["histories"]]
        curves_loss = [h["train_loss"] for h in r["histories"]]
        m_acc, s_acc = _mean_curve(curves_acc)
        m_loss, s_loss = _mean_curve(curves_loss)
        x = range(1, len(m_acc) + 1)
        ax1.plot(x, m_acc, label=f"{label} ({r['mean_acc']:.3f}±{r['std_acc']:.3f})")
        ax1.fill_between(x, m_acc - s_acc, m_acc + s_acc, alpha=0.15)
        ax2.plot(x, m_loss, label=label)
        ax2.fill_between(x, m_loss - s_loss, m_loss + s_loss, alpha=0.15)
    ax1.set_title("Test Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Training Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Loss Function Comparison (n={N_RUNS}, cosine={'on' if USE_COSINE_ANNEALING else 'off'})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_loss_comparison.png", dpi=150)
    plt.close(fig)
    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment 2: Learning Rate Comparison (finer grid, multi-seed)
# ═══════════════════════════════════════════════════════════════════
def exp_learning_rate():
    print("\n" + "=" * 60)
    print("Experiment 2: Learning Rate Comparison (finer grid, multi-seed)")
    print(f"LRs: {EXP_LR_LIST} | Seeds: {SEEDS[:N_RUNS]}")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for lr in EXP_LR_LIST:
        print(f"\n--- LR={lr} ---")
        accs, times = [], []

        def _run(seed):
            train_loader, test_loader = get_dataloaders(DEFAULT_BS, NUM_WORKERS, seed=seed)
            model = _new_model()
            t0 = time.time()
            hist = train(model, train_loader, test_loader,
                         lr=lr, epochs=EXP_EPOCHS, loss_fn=criterion,
                         name=f"exp_lr_{lr}_s{seed}", save_best=False,
                         use_cosine=False, seed=seed)
            elapsed = time.time() - t0
            accs.append(hist["best_acc"])
            times.append(elapsed)
            print(f"  [seed={seed}] best_acc: {hist['best_acc']:.4f} @ {hist['best_epoch']}ep | {elapsed:.0f}s")
            return hist

        histories = _multi_seed_run(_run)
        results[f"LR={lr}"] = {
            "histories": histories,
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_time": np.mean(times),
        }
        print(f"  => Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, r in results.items():
        curves_acc = [h["test_acc"] for h in r["histories"]]
        curves_loss = [h["train_loss"] for h in r["histories"]]
        m_acc, s_acc = _mean_curve(curves_acc)
        m_loss, s_loss = _mean_curve(curves_loss)
        x = range(1, len(m_acc) + 1)
        ax1.plot(x, m_acc, label=f"{label} ({r['mean_acc']:.3f}±{r['std_acc']:.3f})")
        ax1.fill_between(x, m_acc - s_acc, m_acc + s_acc, alpha=0.15)
        ax2.plot(x, m_loss, label=label)
        ax2.fill_between(x, m_loss - s_loss, m_loss + s_loss, alpha=0.15)
    ax1.set_title("Test Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Training Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Learning Rate Comparison (n={N_RUNS})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_lr_comparison.png", dpi=150)
    plt.close(fig)
    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment 3: Batch Size Comparison (multi-seed)
# ═══════════════════════════════════════════════════════════════════
def exp_batch_size():
    print("\n" + "=" * 60)
    print("Experiment 3: Batch Size Comparison (multi-seed)")
    print(f"BS: {EXP_BS_LIST} | Seeds: {SEEDS[:N_RUNS]}")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for bs in EXP_BS_LIST:
        print(f"\n--- Batch={bs} ---")
        accs, times = [], []

        def _run(seed):
            train_loader, test_loader = get_dataloaders(bs, NUM_WORKERS, seed=seed)
            model = _new_model()
            t0 = time.time()
            hist = train(model, train_loader, test_loader,
                         lr=DEFAULT_LR, epochs=EXP_EPOCHS, loss_fn=criterion,
                         name=f"exp_bs_{bs}_s{seed}", save_best=False,
                         use_cosine=False, seed=seed)
            elapsed = time.time() - t0
            accs.append(hist["best_acc"])
            times.append(elapsed)
            print(f"  [seed={seed}] best_acc: {hist['best_acc']:.4f} @ {hist['best_epoch']}ep | "
                  f"{elapsed/hist['stopped_epoch']:.0f}s/ep | {elapsed:.0f}s total")
            return hist

        histories = _multi_seed_run(_run)
        results[f"BS={bs}"] = {
            "histories": histories,
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_time": np.mean(times), "avg_epoch_time": np.mean(times) / histories[0]["stopped_epoch"],
        }
        print(f"  => Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, r in results.items():
        curves_acc = [h["test_acc"] for h in r["histories"]]
        curves_loss = [h["train_loss"] for h in r["histories"]]
        m_acc, s_acc = _mean_curve(curves_acc)
        m_loss, s_loss = _mean_curve(curves_loss)
        x = range(1, len(m_acc) + 1)
        ax1.plot(x, m_acc, label=f"{label} ({r['mean_acc']:.3f}±{r['std_acc']:.3f})")
        ax1.fill_between(x, m_acc - s_acc, m_acc + s_acc, alpha=0.15)
        ax2.plot(x, m_loss, label=label)
        ax2.fill_between(x, m_loss - s_loss, m_loss + s_loss, alpha=0.15)
    ax1.set_title("Test Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Training Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Batch Size Comparison (n={N_RUNS})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_bs_comparison.png", dpi=150)
    plt.close(fig)
    return results


# ═══════════════════════════════════════════════════════════════════
# Experiment 4: SE Ablation (multi-seed)
# ═══════════════════════════════════════════════════════════════════
def exp_se_ablation():
    print("\n" + "=" * 60)
    print("Experiment 4: SE Attention Ablation (multi-seed)")
    print(f"Seeds: {SEEDS[:N_RUNS]}")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for name in ["with_SE", "no_SE"]:
        print(f"\n--- {name} ---")
        accs, times = [], []

        def _run(seed):
            train_loader, test_loader = get_dataloaders(DEFAULT_BS, NUM_WORKERS, seed=seed)
            model = _new_model()
            if name == "no_SE":
                model.se = nn.Identity()
            t0 = time.time()
            hist = train(model, train_loader, test_loader,
                         lr=DEFAULT_LR, epochs=50, loss_fn=criterion,
                         name=f"exp_se_{name}_s{seed}", save_best=(seed == SEEDS[0]),
                         use_cosine=USE_COSINE_ANNEALING, seed=seed)
            elapsed = time.time() - t0
            accs.append(hist["best_acc"])
            times.append(elapsed)
            print(f"  [seed={seed}] best_acc: {hist['best_acc']:.4f} @ {hist['best_epoch']}ep | "
                  f"stop: {hist['stopped_epoch']}ep | {elapsed:.0f}s")
            return hist

        histories = _multi_seed_run(_run)
        results[name] = {
            "histories": histories,
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_time": np.mean(times),
        }
        print(f"  => Mean ± Std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, r in results.items():
        curves_acc = [h["test_acc"] for h in r["histories"]]
        curves_loss = [h["train_loss"] for h in r["histories"]]
        m_acc, s_acc = _mean_curve(curves_acc)
        m_loss, s_loss = _mean_curve(curves_loss)
        x = range(1, len(m_acc) + 1)
        ax1.plot(x, m_acc, label=f"{label} ({r['mean_acc']:.3f}±{r['std_acc']:.3f})")
        ax1.fill_between(x, m_acc - s_acc, m_acc + s_acc, alpha=0.15)
        ax2.plot(x, m_loss, label=label)
        ax2.fill_between(x, m_loss - s_loss, m_loss + s_loss, alpha=0.15)
    ax1.set_title("Test Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Training Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"SE Attention Ablation (n={N_RUNS}, cosine={'on' if USE_COSINE_ANNEALING else 'off'})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp_se_ablation.png", dpi=150)
    plt.close(fig)
    return results


# ═══════════════════════════════════════════════════════════════════
# Master runner
# ═══════════════════════════════════════════════════════════════════
def run_all_experiments():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    all_results = {}
    all_results["loss"]       = exp_loss_function()
    all_results["lr"]         = exp_learning_rate()
    all_results["bs"]         = exp_batch_size()
    all_results["se_ablation"] = exp_se_ablation()

    # Save summary CSV with mean ± std
    with open(RESULTS_DIR / "experiments_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "config", "mean_acc", "std_acc", "mean_time_s", "n_runs"])
        for exp_name, exp_data in all_results.items():
            for cfg, v in exp_data.items():
                w.writerow([
                    exp_name, cfg, f"{v['mean_acc']:.4f}", f"{v['std_acc']:.4f}",
                    f"{v['mean_time']:.0f}", N_RUNS,
                ])

    print("\n" + "=" * 60)
    print("All experiments completed. Summary saved to", RESULTS_DIR / "experiments_summary.csv")
    return all_results
