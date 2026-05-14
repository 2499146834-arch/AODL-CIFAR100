# AODL — CIFAR-100 Image Classification with ResNet-SE

[![Accuracy](https://img.shields.io/badge/Accuracy-75.66%25-brightgreen)](https://github.com/2499146834-arch/AODL-CIFAR100)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti-blue)]()

Reproduction and improvement of the [AODL project](https://github.com/951135150/CDS525-Group-Project-ResNetSE) (CDS 525). Achieves **75.66%** test accuracy on CIFAR-100, exceeding the [original](https://github.com/951135150/CDS525-Group-Project-ResNetSE) **72%** by **+3.66%** through cosine annealing, label smoothing, and mixed precision training.

## Key Improvements

| | Original | This Work |
|---|---|---|
| **Accuracy** | 72% | **75.66%** |
| LR Schedule | Fixed | Cosine Annealing |
| Loss | CELoss only | + Label Smoothing (0.1) |
| Precision | FP32 | AMP (Mixed Precision) |
| Reproducibility | Single run | 3 seeds, mean ± std |
| SE Ablation | Not performed | Included |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python main.py train

# Run experiments
python main.py exp

# Evaluate
python main.py test
```

## Web Demo

```bash
python app.py
# Open http://127.0.0.1:5000
```

Upload any image to get top-5 CIFAR-100 predictions with confidence scores.

## Project Structure

```
├── main.py              # CLI entry point (train / exp / test)
├── app.py               # Flask web demo
├── aodl/                # Core Python package
│   ├── __init__.py
│   ├── config.py        # Centralized hyperparameters & paths
│   ├── model.py         # ResNet-18 + SE attention (CIFAR100ResNet)
│   ├── data.py          # CIFAR-100 data loaders & augmentations
│   ├── train.py         # Training loop (AMP, early stopping, cosine annealing)
│   ├── eval.py          # Evaluation (classification report, confusion matrix, ROC)
│   └── experiments.py   # Multi-seed comparative experiments (LR, BS, loss, SE)
├── models/              # Model artifacts
│   ├── checkpoints/     # Trained model weights (.pth)
│   └── pretrained/      # Pretrained ResNet-18 backbone
├── outputs/             # Generated outputs
│   ├── figures/         # Plots (.png)
│   └── results/         # Reports & experiment CSV
├── data/                # CIFAR-100 dataset (auto-downloaded)
├── docs/                # Project reports (CN + EN)
├── requirements.txt     # Python dependencies
└── .gitignore
```

All source modules are under the `aodl` package. Scripts at root import via `from aodl.xxx import ...`.

## Results Summary

| Experiment | Best Config | Accuracy |
|---|---|---|
| Loss Function | Label Smoothing (0.1) | **75.36%** ± 0.27 |
| Learning Rate | 5e-5 | 73.21% ± 0.23 |
| Batch Size | 128 | 72.52% ± 0.18 |
| SE Ablation | No significant gain | 74.50% vs 74.55% |

## Final Model Performance

| Metric | Value |
|---|---|
| Accuracy | 75.66% |
| Macro F1 | 75% |
| ROC AUC (micro) | 0.9867 |

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| torch >= 2.0 | Deep learning framework |
| torchvision >= 0.15 | Pretrained models & datasets |
| scikit-learn | Metrics (classification report, confusion matrix, ROC) |
| matplotlib | Figure generation |
| numpy | Numerical computation |
| flask | Web demo (`app.py`) |
| pillow | Image I/O for web demo |

## Reference

- Original project: [CDS525-Group-Project-ResNetSE](https://github.com/951135150/CDS525-Group-Project-ResNetSE)
- This reproduction: [AODL-CIFAR100](https://github.com/2499146834-arch/AODL-CIFAR100)
