import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from typing import List, Callable, Any, Type
import pickle as pkl
import os
import gc

from lenet import LeNet
from dataset import get_CIFAR
from traineval import train, evaluate
from singletraining import train_single_models
from ensemble import dynamic_ensemble_cifar, generated_trained_samples
from generate import generate_random_rgb_image
from student import train_student, evaluate_student, TransformerConvNet

# https://machinelearningmastery.com/dynamic-ensemble-selection-in-python/

if __name__ == '__main__':
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    """train_single_models(LeNet, 8, transform, epochs=1000)
    torch.cuda.empty_cache()
    gc.collect()"""

    """dynamic_ensemble_cifar(8, transform, 4)
    torch.cuda.empty_cache()
    gc.collect()"""

    """g_model = train_student()
    torch.cuda.empty_cache()
    gc.collect()"""
    """g_model = TransformerConvNet(3, 256, 10, 4, 2, 3).to('cuda')
    g_model.load_state_dict(torch.load(f'generator.pt'))
    evaluate_student(g_model, transform, epochs=1000)"""
    #generate_random_rgb_image(50000, transform)
    print('\n\n\n\nGENERATED 50000 RANDOM IMAGES')
    #generated_trained_samples(transform, 8, 4)
    g_model = train_student()
    torch.cuda.empty_cache()
    gc.collect()
    g_model = g_model.to('cuda')
    evaluate_student(g_model, transform, epochs=1000)
