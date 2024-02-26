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


def train_eval_loop(generator, classifiers, data_loader, input_size_G, knn, k, pairs, device,
                    criterion_G, optimizer_G, batch_size, mode):
    total_loss = list()
    corrects = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        input_G = torch.randn(1, input_size_G).to(device)
        classifiers_weights = generator(input_G).cpu()

        # Assign the fake weights to the Classifiers
        for i, classifier in enumerate(classifiers):
            classifier_state_dict = classifier.state_dict()
            with torch.no_grad():
                for name, param in classifier_state_dict.items():
                    param.copy_(classifiers_weights[:, :, i].view(-1)[0:param.numel()].view(param.shape))

        # Perform dynamic ensemble
        losses = []
        outputs_list = list()
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
            outputs_list.append(outputs)
            _, predicted = torch.max(outputs, 1)
            #print(predicted)
            #print(label.to(device))
            #losses.append(criterion_G(predicted, torch.tensor([label.item()], device='cuda')))

            if predicted.detach() == label:
                corrects += 1

        # TODO: from outputs list get tensor for computing the loss
        tensor_outputs = torch.tensor(outputs_list, device='cuda:0')
        ensemble_loss = criterion_G(tensor_outputs, labels)
        # Get loss from dynamic ensemble
        #ensemble_loss = torch.mean(torch.tensor(losses))
        # Add loss to total loss
        total_loss.append(ensemble_loss.item())
        # optimize
        if mode == 'training':
            optimizer_G.zero_grad()
            ensemble_loss.backward()
            optimizer_G.step()

    avg_loss = np.mean(total_loss)
    accuracy = corrects / (len(data_loader) * batch_size)
    return avg_loss, accuracy
