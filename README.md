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
├── main.py              # Entry point (train / exp / test)
├── model.py             # ResNet-18 + SE attention
├── data.py              # CIFAR-100 data pipeline
├── train.py             # Training loop + AMP + cosine annealing
├── eval.py              # Evaluation, confusion matrix, ROC, visualizations
├── experiments.py       # Multi-seed comparative experiments
├── app.py               # Flask web demo
├── config.py            # Hyperparameters
├── checkpoints/         # Trained model weights
├── figures/             # Generated plots
├── results/             # Classification report + experiment CSV
└── reports/             # Full experiment reports (CN + EN)
```

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

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- Flask (for web demo)

## Reference

- Original project: [CDS525-Group-Project-ResNetSE](https://github.com/951135150/CDS525-Group-Project-ResNetSE)
- This reproduction: [AODL-CIFAR100](https://github.com/2499146834-arch/AODL-CIFAR100)
