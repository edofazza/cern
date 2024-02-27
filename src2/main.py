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
from perform import eval_loop, train_loop

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 64
    train_loader, val_loader, test_loader = get_CIFAR(transform, batch_size=batch_size)

    # Define Generator
    #   the generator has an output equal to the number of parameters
    #   times how many classifier are present
    output_size_G = sum(p.numel() for p in LeNet().parameters())
    num_classifiers = 2
    input_size_G = output_size_G * num_classifiers
    generator = Generator(input_size_G, output_size_G, num_classifiers).to(device)

    # Define F
    #   LeNet for classification
    classifiers = [LeNet().to(device) for _ in range(num_classifiers)]
    for classifier in classifiers:
        classifier.eval()

    # Define optimizer for Generator
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    criterion_G = nn.CrossEntropyLoss()

    # samples for KNN
    pairs = []
    k = 3
    for inputs, labels in train_loader:
        for input, label in zip(inputs, labels):
            pairs.append((input.reshape(input.shape[0] * input.shape[1] * input.shape[2]), label))
    samples = [pair[0] for pair in pairs]
    if os.path.exists('knn.pkl'):
        with open("knn.pkl", "rb") as f:
            knn = pkl.load(f)
    else:
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(samples)
        with open("knn.pkl", "wb") as f:
            pkl.dump(knn, f)

    epochs = 1000
    early_stopping_threshold = 20
    early_stopping_counter = 0
    best_weights_G = generator.state_dict()
    best_loss = 100000
    for epoch in range(epochs):
        # Training loop
        generator.train()
        avg_loss, accuracy = train_loop(generator, classifiers, train_loader, input_size_G,
                                        knn, k, pairs, device, criterion_G, optimizer_G, batch_size,
                                        'training')
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

        # Validation Loop
        generator.eval()
        with torch.no_grad():
            avg_loss, accuracy = eval_loop(classifiers, test_loader, knn, k, pairs, criterion_G, device, batch_size)
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

        if best_loss > avg_loss:
            best_loss = avg_loss
            early_stopping_counter += 1
            best_weights_G = generator.state_dict()
            if early_stopping_threshold == early_stopping_counter:
                generator.load_state_dict(best_weights_G)
                break

    for i, classifier in enumerate(classifiers):
        torch.save(classifier.state_dict(), f'new/model{i}.pt')

    # Testing Loop
    avg_loss, accuracy = eval_loop(classifiers, test_loader, knn, k, pairs, criterion_G, device, batch_size)
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
