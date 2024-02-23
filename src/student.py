import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from typing import Callable, Type

from dataset import get_CIFAR
from singletraining import evaluate


class TransformerConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, kernel_size):
        super(TransformerConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()

        # Transformer layers
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layers,
            num_layers=num_layers
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        x = self.conv1(x)
        x = self.relu(x)

        # Reshape for transformer input
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)

        # Transformer forward pass
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=0)

        # Fully connected layer
        x = self.fc(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.samples = self.load_samples()

    def load_samples(self):
        samples = []
        files = [f for f in os.listdir(self.directory) if f.endswith('.npy')]
        for file in files:
            filepath = os.path.join(self.directory, file)
            data = np.load(filepath, allow_pickle=True)
            filepath = os.path.join(self.directory + '_label', file)
            label = np.load(filepath, allow_pickle=True)
            samples.append((data, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_data = torch.tensor(sample[0], dtype=torch.float32)
        label = torch.tensor(sample[1], dtype=torch.float32)
        return input_data, label


def train_student(input_dim: int = 3, hidden_dim: int = 256, output_dim: int = 10,
                  num_heads: int = 4, num_layers: int = 2, kernel_size: int = 3,
                  epochs: int = 1000):
    print('\n\n\n\n\n\n\nTRAINING G NETWORK')
    model = TransformerConvNet(input_dim, hidden_dim, output_dim, num_heads, num_layers, kernel_size).to('cuda')

    training_directory = 'training_and_generated'
    validation_directory = 'validation_and_generated'

    # Create datasets and data loaders
    train_dataset = CustomDataset(training_directory)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    val_dataset = CustomDataset(validation_directory)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_average_val_loss = 1000000
    early_stopping_counter = 0
    best_model_dict = model.state_dict().copy()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        average_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        average_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_train_loss}, Validation Loss: {average_val_loss:.4f}")

        if best_average_val_loss > average_val_loss:
            best_average_val_loss = average_val_loss
            best_model_dict = model.state_dict().copy()
        else:
            early_stopping_counter += 1
            if early_stopping_counter > 20:
                print('EARLY STOPPING, 20 epochs without improving')
                break

    torch.save(best_model_dict, f'generator.pt')
    model.load_state_dict(best_model_dict)
    return model


def evaluate_student(g_model: nn.Module, transform: Callable,
         batch_size: int = 256, learning_rate: float = 0.001, epochs: int = 10,
         train_val_split: float = 0.8, device: str = 'cuda') -> None:
    assert 0.0 < train_val_split < 1.0
    assert epochs > 0
    assert learning_rate > 0
    assert batch_size > 0
    print('\n\n\n\n\n\n\nG NETWORK RESULTS:')
    train_loader, val_loader, test_loader = get_CIFAR(transform, batch_size, train_val_split)
    criterion = nn.CrossEntropyLoss()
    evaluate([g_model], train_loader, criterion, 0, 0, 'Training', device)
    evaluate([g_model], train_loader, criterion, 0, 0, 'Val', device)
    evaluate([g_model], train_loader, criterion, 0, 0, 'Test', device)
