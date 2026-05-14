"""All hyperparameters and configuration centralized here."""

import torch
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────
SEEDS = [42, 123, 999]  # multi-seed for statistical reliability

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
WEIGHTS_DIR = ROOT / "models" / "pretrained"

# Redirect all torch cache to project dir (don't use C:)
torch.hub.set_dir(str(WEIGHTS_DIR))
import os
os.environ["TORCH_HOME"] = str(WEIGHTS_DIR)

# ── Device ───────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset ──────────────────────────────────────────────────────
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGE_SIZE    = 64
NUM_CLASSES   = 100

# ── Model Architecture ───────────────────────────────────────────
SE_REDUCTION  = 16
DROPOUT       = 0.4
HIDDEN_DIM    = 256
BACKBONE_OUT  = 512  # ResNet-18 fc.in_features

# ── Training Defaults ────────────────────────────────────────────
DEFAULT_LR     = 1e-4
DEFAULT_BS     = 128
DEFAULT_EPOCHS = 50
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOP_PATIENCE = 15
NUM_WORKERS    = 4

# ── Mixed Precision (AMP) ────────────────────────────────────────
USE_AMP = True  # 1.5-2x speedup, half VRAM

# ── Cosine Annealing (improvement over fixed LR) ─────────────────
USE_COSINE_ANNEALING = True
COSINE_T_MAX         = 50      # period of cosine cycle
COSINE_ETA_MIN       = 1e-6    # minimum LR

# ── Experiment Configs ───────────────────────────────────────────
EXP_LR_LIST    = [5e-5, 1e-4, 2e-4, 5e-4]  # finer grid around optimal
EXP_BS_LIST    = [16, 32, 64, 128]
EXP_EPOCHS     = 20  # shorter for comparison experiments
N_RUNS         = 3   # runs per config for mean ± std
