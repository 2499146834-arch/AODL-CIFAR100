"""Full evaluation: classification report, confusion matrix, ROC, visualizations."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize
from config import DEVICE, NUM_CLASSES, RESULTS_DIR, FIGURES_DIR


def full_evaluation(model, test_loader, class_names):
    """Run full evaluation and save all artifacts."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_logits = np.concatenate(all_logits)

    # ── Classification Report ──────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3,
    )
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    acc = (y_pred == y_true).mean()
    print(f"Test Accuracy: {acc:.4f}")
    print(report[:500])

    # ── Confusion Matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — CIFAR-100 (Acc: {acc:.2%})")
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── ROC Curve (micro-average) ──────────────────────────────────
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    y_probs = torch.softmax(torch.tensor(y_logits), dim=1).numpy()

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                    estimator_name="ResNet-SE").plot(ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"Micro-Average ROC — CIFAR-100 (AUC={roc_auc:.3f})")
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC AUC (micro): {roc_auc:.4f}")
    return {"accuracy": acc, "roc_auc": roc_auc, "report": report}


def visualize_predictions(model, test_loader, class_names, num_samples=100):
    """Grid visualization of predictions (green=correct, red=wrong)."""
    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x_dev = x.to(DEVICE)
            logits = model(x_dev)
            p = logits.argmax(1).cpu()
            # gather up to num_samples
            for i in range(x.size(0)):
                if len(images) >= num_samples:
                    break
                images.append(x[i])
                labels.append(y[i].item())
                preds.append(p[i].item())
            if len(images) >= num_samples:
                break

    images = images[:num_samples]
    labels = labels[:num_samples]
    preds = preds[:num_samples]

    cols = 10
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = axes.flatten()

    mean = np.array([0.5071, 0.4867, 0.4408])
    std  = np.array([0.2675, 0.2565, 0.2761])

    for i, ax in enumerate(axes):
        if i >= num_samples:
            ax.axis("off")
            continue
        img = images[i].permute(1, 2, 0).numpy()
        img = img * std + mean  # denormalize
        img = np.clip(img, 0, 1)

        color = "green" if preds[i] == labels[i] else "red"
        title = f"P:{class_names[preds[i]][:6]}\nA:{class_names[labels[i]][:6]}"
        ax.imshow(img)
        ax.set_title(title, fontsize=5, color=color, pad=1)
        ax.axis("off")

    fig.suptitle("Predictions — green=correct, red=incorrect", fontsize=10, y=0.92)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction visualization ({num_samples} samples).")
