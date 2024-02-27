import torch
from torch import optim
from torch import nn
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import pickle as pkl
import os

import numpy as np
from typing import List

from dataset import get_CIFAR
from lenet import LeNet
from generator import Generator


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleList(models)

    def forward(self, x) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


def train_loop(generator, classifiers, data_loader, input_size_G, knn, k, pairs, device,
               criterion_G, optimizer_G, batch_size, mode):
    corrects = 0
    losses = 0
    input_G = torch.randn(1, input_size_G).to(device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        classifiers_weights = generator(input_G).cpu()

        # Assign the fake weights to the Classifiers
        for i, classifier in enumerate(classifiers):
            classifier_state_dict = classifier.state_dict()
            with torch.no_grad():
                for name, param in classifier_state_dict.items():
                    param.copy_(classifiers_weights[:, :, i].view(-1)[0:param.numel()].view(param.shape))
        input_G = classifiers_weights.view(-1, input_size_G).to(device)

        # Perform dynamic ensemble
        for input, label in zip(inputs, labels):
            input_numpy = input.detach().cpu().numpy()
            input_numpy = input_numpy.reshape(1, input_numpy.shape[0] * input_numpy.shape[1] * input_numpy.shape[2])
            indices = knn.kneighbors(input_numpy, return_distance=False)[0]

            j = 0
            while True:
                selected_models = []
                if mode == 'training':
                    tmp_indices = indices[1: k + 1 - j]
                else:
                    tmp_indices = indices[0: k - j]
                # create a batch of those sample
                closest_pairs_input = [pairs[idx][0].reshape(3, 32, 32) for idx in tmp_indices]
                closest_pairs_label = [pairs[idx][1] for idx in tmp_indices]
                # evaluate models
                batched_pairs = torch.from_numpy(np.array(closest_pairs_input, dtype=np.float32)).to('cuda')

                for model in classifiers:
                    model = model.to('cuda')
                    model.eval()
                    outputs = model(batched_pairs)
                    _, predicted = torch.max(outputs, 1)
                    closest_pairs_label = torch.from_numpy(
                        np.array(closest_pairs_label).reshape(len(closest_pairs_label)))
                    total_correct = (predicted.detach().cpu() == closest_pairs_label).sum().item()
                    # select only the models that can classify correctly all kNN
                    if total_correct == len(closest_pairs_input):
                        selected_models.append(model)

                if not selected_models:
                    j += 1
                    if k + 1 - j == 1 and mode == 'training':
                        selected_models.append(classifiers[0])
                        break
                    elif k - j == 1 and not mode == 'training':
                        selected_models.append(classifiers[0])
                        break
                else:
                    break
            ensemble = EnsembleModel(selected_models).to(device)
            outputs = ensemble(input.to(device))
            _, predicted = torch.max(outputs, 1)
            loss = criterion_G(outputs, torch.tensor([label.item()], device='cuda'))

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            losses += loss.item()

            if predicted.detach() == label:
                corrects += 1

    avg_loss = losses / (len(data_loader) * batch_size)
    accuracy = corrects / (len(data_loader) * batch_size)
    return avg_loss, accuracy


def eval_loop(classifiers, data_loader, knn, k, pairs, criterion_G, device, batch_size):
    corrects = 0
    losses = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Perform dynamic ensemble
        for input, label in zip(inputs, labels):
            input_numpy = input.detach().cpu().numpy()
            input_numpy = input_numpy.reshape(1, input_numpy.shape[0] * input_numpy.shape[1] * input_numpy.shape[2])
            indices = knn.kneighbors(input_numpy, return_distance=False)[0]

            j = 0
            while True:
                selected_models = []
                tmp_indices = indices[0: k - j]
                # create a batch of those sample
                closest_pairs_input = [pairs[idx][0].reshape(3, 32, 32) for idx in tmp_indices]
                closest_pairs_label = [pairs[idx][1] for idx in tmp_indices]
                # evaluate models
                batched_pairs = torch.from_numpy(np.array(closest_pairs_input, dtype=np.float32)).to('cuda')

                for model in classifiers:
                    model = model.to('cuda')
                    model.eval()
                    outputs = model(batched_pairs)
                    _, predicted = torch.max(outputs, 1)
                    closest_pairs_label = torch.from_numpy(
                        np.array(closest_pairs_label).reshape(len(closest_pairs_label)))
                    total_correct = (predicted.detach().cpu() == closest_pairs_label).sum().item()
                    # select only the models that can classify correctly all kNN
                    if total_correct == len(closest_pairs_input):
                        selected_models.append(model)

                if not selected_models:
                    j += 1
                    if k - j == 1:
                        selected_models.append(classifiers[0])
                        break
                else:
                    break
            ensemble = EnsembleModel(selected_models).to(device)
            outputs = ensemble(input.to(device))
            _, predicted = torch.max(outputs, 1)
            loss = criterion_G(outputs, torch.tensor([label.item()], device='cuda'))

            losses += loss.item()

            if predicted.detach() == label:
                corrects += 1

    avg_loss = losses / (len(data_loader) * batch_size)
    accuracy = corrects / (len(data_loader) * batch_size)
    return avg_loss, accuracy
