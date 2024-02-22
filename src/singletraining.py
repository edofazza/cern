import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torchvision
from typing import List, Callable, Any, Type

from dataset import get_CIFAR
from traineval import train, evaluate


def train_single_models(f_model: Type[nn.Module], n: int, transform: Callable,
         batch_size: int = 64, learning_rate: float = 0.001, epochs: int = 10,
         train_val_split: float = 0.8, device: str = 'cuda') -> None:
    assert 0.0 < train_val_split < 1.0
    assert epochs > 0
    assert learning_rate > 0
    assert n > 1
    assert batch_size > 0
    train_loader, val_loader, test_loader = get_CIFAR(transform, batch_size, train_val_split)

    # Initialize models and optimizers
    models = [f_model().to(device) for _ in range(n)]
    criterion = nn.CrossEntropyLoss()

    # Training-Validation loop
    best_overall_average_loss = 100
    early_stopping_counter = 0
    best_models = None

    for epoch in range(epochs):
        train(models, train_loader, learning_rate, criterion, epoch, epochs, device)
        overall_average_loss = evaluate(models, val_loader, criterion, epoch, epochs, 'Validation', device)

        # Early stopping
        if best_overall_average_loss < overall_average_loss:
            early_stopping_counter += 1
            if early_stopping_counter > 5:
                break
        else:
            best_overall_average_loss = overall_average_loss
            best_models = models.copy()

    # Save models
    for i, model in enumerate(best_models):
        torch.save(model.state_dict(), f'model_{i}.pt')

    # Evaluate on test set
    evaluate(best_models, test_loader, criterion, 0, 0, 'Test', device)
