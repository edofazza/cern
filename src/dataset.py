from torch.utils.data import DataLoader, random_split
import torchvision
import torch
from typing import Callable, Any


def get_CIFAR(transform: Callable, batch_size: int = 64, train_val_split: float = 0.8) \
        -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # Load CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(train_val_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
