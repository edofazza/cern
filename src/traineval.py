import torch
import torch.optim as optim
import numpy as np


def train(models, train_loader, learning_rate, criterion, epoch, epochs, device):
    for i, model in enumerate(models):
        model.train()
        total_correct_train = 0
        total_samples_train = 0
        total_loss_train = 0.0
        optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_model.step()

            total_loss_train += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples_train += labels.size(0)
            total_correct_train += (predicted == labels).sum().item()

        avg_loss_train = total_loss_train / len(train_loader)
        train_accuracy = total_correct_train / total_samples_train
        print(f'Model {i}, Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss_train:.4f}, Training Accuracy: {train_accuracy:.4f}')


def evaluate(models, val_loader, criterion, epoch, epochs, label, device):
    overall_loss = list()
    for i, model in enumerate(models):
        model.eval()
        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_samples_val += labels.size(0)
                total_correct_val += (predicted == labels).sum().item()

            # Average validation loss and accuracy for the current model
            avg_loss_val = total_loss_val / len(val_loader)
            val_accuracy = total_correct_val / total_samples_val
            overall_loss.append(avg_loss_val)
            if label == 'Validation':
                print(f'Model {i}, Epoch {epoch + 1}/{epochs}, {label} Loss: {avg_loss_val:.4f}, {label} Accuracy: {val_accuracy:.4f}')
            else:
                print(
                    f'Model {i}, {label} Loss: {avg_loss_val:.4f}, {label} Accuracy: {val_accuracy:.4f}')
    return np.mean(overall_loss)