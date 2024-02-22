import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torchvision
from typing import List, Callable, Any, Type
import pickle as pkl
import os

from lenet import LeNet
from dataset import get_CIFAR
from traineval import train, evaluate
from singletraining import train_single_models
from ensemble import dynamic_ensemble_cifar
from generate import generate_random_rgb_image
from student import train_student

# https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/

if __name__ == '__main__':
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_single_models(LeNet, 8, transform, epochs=500)

    dynamic_ensemble_cifar(8, transform, 4)

    #generate_random_rgb_image(100000, transform)
    #generated_trained_samples(pairs, 8, 4)

    train_student()

