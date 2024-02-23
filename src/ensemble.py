import torch
import torch.nn as nn
from typing import List
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle as pkl
import uuid
import os
import gc

from dataset import get_CIFAR
from lenet import LeNet


class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleList(models)

    def forward(self, x) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


def collect_and_analyze_ensemble_outputs(models, knn, loader, k, pairs, mode='training'):
    corrects = 0
    for inputs, labels in loader:
        for input, label in zip(inputs, labels):
            input_numpy = input.numpy()
            input_numpy = input_numpy.reshape(1, input_numpy.shape[0] * input_numpy.shape[1] * input_numpy.shape[2])
            # select kNN of input
            indices = knn.kneighbors(input_numpy, return_distance=False)[0]
            j = 0
            while True:
                if mode == 'training':
                    tmp_indices = indices[1: k + 1 - j]
                else:
                    tmp_indices = indices[0: k - j]
                # create a batch of those sample
                closest_pairs_input = [pairs[idx][0].reshape(3, 32, 32) for idx in tmp_indices]
                closest_pairs_label = [pairs[idx][1] for idx in tmp_indices]
                #print(np.array(closest_pairs_input, dtype=np.float32).shape)
                # evaluate models
                batched_pairs = torch.from_numpy(np.array(closest_pairs_input, dtype=np.float32)).to('cuda')
                selected_models = []
                for model in models:
                    model = model.to('cuda')
                    model.eval()
                    outputs = model(batched_pairs)
                    _, predicted = torch.max(outputs, 1)
                    closest_pairs_label = torch.from_numpy(np.array(closest_pairs_label).reshape(len(closest_pairs_label)))
                    total_correct = (predicted.detach().cpu() == closest_pairs_label).sum().item()
                    # select only the models that can classify correctly all kNN
                    if total_correct == len(closest_pairs_input):
                        selected_models.append(model)

                if not selected_models:
                    j += 1
                    if k + 1 - j == 1 and mode == 'training':
                        selected_models.append(models[0])
                        break
                    elif k - j == 1 and not mode == 'training':
                        selected_models.append(models[0])
                        break
                else:
                    break

            # create ensemble
            ensemble = EnsembleModel(selected_models).to('cuda')
            # classify with the ensemble
            outputs = ensemble(input.to('cuda'))
            _, predicted = torch.max(outputs, 1)

            del ensemble
            gc.collect()
            torch.cuda.empty_cache()

            if predicted.detach().cpu() == label:
                corrects += 1
            uid = str(uuid.uuid4())[:16]
            np.save(f'{mode}/{uid}.npy', input.detach().cpu().numpy())
            np.save(f'{mode}_label/{uid}.npy', outputs[0].detach().cpu().numpy())
    if mode == 'training':
        print(f'\t- {mode} accuracy: {corrects / (len(loader) * 16)}')
    else:
        print(f'\t- {mode} accuracy: {corrects/len(loader)}')


def dynamic_ensemble_cifar(n, transform, k):
    # get CIFAR test
    train_loader, _, _ = get_CIFAR(transform, batch_size=16)
    gc.collect()

    # get models
    models = list()
    for i in range(n):
        model = LeNet()
        model.load_state_dict(torch.load(f'model_{i}.pt'))
        models.append(model)

    pairs = []
    for inputs, labels in train_loader:
        for input, label in zip(inputs, labels):
            pairs.append((input.reshape(input.shape[0] * input.shape[1] * input.shape[2]), label))
    # Create kNN
    samples = [pair[0] for pair in pairs]
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(samples)
    with open("knn.pkl", "wb") as f:
        pkl.dump(knn, f)
    #with open("knn.pkl", "rb") as f:
    #    knn = pkl.load(f)
    print('\n\n\n\n\n\nDYNAMIC ENSEMBLE PERFORMANCE:')
    if not os.path.exists('training'):
        os.mkdir('training')
        os.mkdir('validation')
        os.mkdir('test')
        os.mkdir('training_label')
        os.mkdir('validation_label')
        os.mkdir('test_label')
    collect_and_analyze_ensemble_outputs(models, knn, train_loader, k, pairs, 'training')
    del train_loader
    gc.collect()
    torch.cuda.empty_cache()
    _, val_loader, _ = get_CIFAR(transform, batch_size=1)
    collect_and_analyze_ensemble_outputs(models, knn, val_loader, k, pairs, 'validation')
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()
    _, _, test_loader = get_CIFAR(transform, batch_size=1)
    collect_and_analyze_ensemble_outputs(models, knn, test_loader, k, pairs, 'test')
    #return pairs


def generated_trained_samples(transform, n, k):
    train_loader, _, _ = get_CIFAR(transform, batch_size=16)
    gc.collect()

    pairs = []
    for inputs, labels in train_loader:
        for input, label in zip(inputs, labels):
            pairs.append((input.reshape(input.shape[0] * input.shape[1] * input.shape[2]), label))

    os.mkdir('generated_label')
    with open("knn.pkl", "rb") as f:
        knn = pkl.load(f)

    # get models
    models = list()
    for i in range(n):
        model = LeNet()
        model.load_state_dict(torch.load(f'model_{i}.pt'))
        models.append(model)

    samples_path = [s_path for s_path in os.listdir('generated') if s_path.endswith('.npy')]
    for sample in samples_path:
        sample = np.load(f'generated/{sample}')
        flatted_sample = sample.reshape(1, sample.shape[0] * sample.shape[1] * sample.shape[2])
        indices = knn.kneighbors(flatted_sample, return_distance=False)[0]

        j = 0
        while True:
            tmp_indices = indices[0:-j]
            # create a batch of those sample
            closest_pairs_input = [pairs[idx][0].reshape(3, 32, 32) for idx in tmp_indices]
            closest_pairs_label = [pairs[idx][1] for idx in tmp_indices]
            # evaluate models
            batched_pairs = torch.from_numpy(np.array(closest_pairs_input)).to('cuda')
            selected_models = []
            for model in models:
                model = model.to('cuda')
                model.eval()
                outputs = model(batched_pairs)
                _, predicted = torch.max(outputs, 1)
                closest_pairs_label = torch.from_numpy(np.array(closest_pairs_label).reshape(len(closest_pairs_label)))
                total_correct = (predicted.detach().cpu() == closest_pairs_label).sum().item()
                # select only the models that can classify correctly all kNN
                if total_correct == len(closest_pairs_input):
                    selected_models.append(model)

            if not selected_models:
                j += 1
                if j == k:
                    selected_models.append(models[0])
            else:
                break

        # create ensemble
        ensemble = EnsembleModel(selected_models).to('cuda')
        # classify with the ensemble
        outputs = ensemble(torch.from_numpy(sample).reshape(-1).to('cuda'))
        np.save(f'generated_label/{sample}', outputs[0].detach().cpu().numpy())

