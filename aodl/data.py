"""CIFAR-100 data loading with augmentations."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .config import DATA_DIR, CIFAR100_MEAN, CIFAR100_STD, IMAGE_SIZE


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])


def get_dataloaders(batch_size: int, num_workers: int = 4, seed: int = 42):
    """Return (train_loader, test_loader) for CIFAR-100."""
    train_ds = datasets.CIFAR100(
        root=DATA_DIR, train=True, download=True,
        transform=get_transforms(train=True),
    )
    test_ds = datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True,
        transform=get_transforms(train=False),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
