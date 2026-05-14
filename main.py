"""AODL — CIFAR-100 Classification with ResNet-SE.  Entry point.

Modes:
  train  — baseline training with default hyperparams, saves best checkpoint.
  exp    — run all comparative experiments (LR, BS, loss, SE ablation).
  test   — load best checkpoint & run full evaluation + visualizations.
"""

# Windows multiprocessing MUST be first, before any other import
import multiprocessing
multiprocessing.freeze_support()

import sys
import torch
import torch.nn as nn
from aodl.data import get_dataloaders
from aodl.model import CIFAR100ResNet
from aodl.train import train, set_seed
from aodl.eval import full_evaluation, visualize_predictions
from aodl.experiments import run_all_experiments
from aodl.config import (
    DEVICE, SEEDS, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR,
    DEFAULT_LR, DEFAULT_BS, DEFAULT_EPOCHS, LABEL_SMOOTHING,
    NUM_WORKERS, USE_COSINE_ANNEALING,
)


CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def mode_train():
    print(f"TRAIN mode — baseline ResNet-SE, LR=1e-4, BS={DEFAULT_BS}, CELoss, cosine={USE_COSINE_ANNEALING}")
    set_seed(SEEDS[0])
    train_loader, test_loader = get_dataloaders(DEFAULT_BS, NUM_WORKERS, seed=SEEDS[0])
    model = CIFAR100ResNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    hist = train(
        model, train_loader, test_loader,
        lr=DEFAULT_LR, epochs=DEFAULT_EPOCHS, loss_fn=criterion,
        name="baseline", save_best=True,
        use_cosine=USE_COSINE_ANNEALING, seed=SEEDS[0],
    )
    print(f"\nBaseline done. Best acc: {hist['best_acc']:.4f} @ epoch {hist['best_epoch']}")
    return model


def mode_exp():
    print("EXP mode — running all comparative experiments")
    run_all_experiments()


def mode_test():
    print("TEST mode — loading checkpoint & full evaluation")
    set_seed(SEEDS[0])
    _, test_loader = get_dataloaders(DEFAULT_BS, NUM_WORKERS)
    model = CIFAR100ResNet().to(DEVICE)

    ckpt = CHECKPOINT_DIR / "baseline_best.pth"
    if not ckpt.exists():
        ckpt = CHECKPOINT_DIR / "exp_loss_CELoss_s42_best.pth"
    if not ckpt.exists():
        print(f"No checkpoint found in {CHECKPOINT_DIR}\nRun 'train' or 'exp' mode first.")
        return

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"Loaded checkpoint: {ckpt}")

    full_evaluation(model, test_loader, CIFAR100_CLASSES)

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from aodl.config import CIFAR100_MEAN, CIFAR100_STD, IMAGE_SIZE, DATA_DIR
    test_ds = datasets.CIFAR100(
        root=DATA_DIR, train=False, download=False,
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]),
    )
    test_loader_viz = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)
    visualize_predictions(model, test_loader_viz, CIFAR100_CLASSES, num_samples=100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|exp|test]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "train":
        mode_train()
    elif mode == "exp":
        mode_exp()
    elif mode == "test":
        mode_test()
    else:
        print(f"Unknown mode: {mode}. Use train, exp, or test.")
        sys.exit(1)
