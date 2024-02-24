"""
    IMPORTS
    Note: used PyTorch for solving the task

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List, Callable, Any, Type
import pickle as pkl
import uuid
import os
import gc

"""
    F Network
        Implemented a LeNet network to solve the task, as suggested 
        in the guidelines
"""
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
    Ensemble model
        Ensemble model to use and create dynamically for 
        implementing the KNORA eliminate approach
"""
class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models = nn.ModuleList(models)

    def forward(self, x) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


"""
    G Network
      A very simple Transformer Convolutional Network  
"""
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
        x = self.conv1(x)
        x = self.relu(x)
        # Reshape for transformer input
        x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)
        # Transformer forward pass
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=0)
        x = self.fc(x)
        return x


"""
    Function for obtaining CIFAR10 dataset using torchvision
    and create DataLoader object for each set (training, validation,
    test)
"""
def get_CIFAR(transform: Callable, batch_size: int = 64, train_val_split: float = 0.8) \
        -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # Load CIFAR-10 dataset
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split the dataset into training and validation sets
    train_size = int(train_val_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                              torch.Generator().manual_seed(42))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


"""
    Functions to train and evaluate (validation, test) the F networks 
"""
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
        print(
            f'Model {i}, Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss_train:.4f}, Training Accuracy: {train_accuracy:.4f}')


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
                print(
                    f'Model {i}, Epoch {epoch + 1}/{epochs}, {label} Loss: {avg_loss_val:.4f}, {label} Accuracy: {val_accuracy:.4f}')
            else:
                print(
                    f'Model {i}, {label} Loss: {avg_loss_val:.4f}, {label} Accuracy: {val_accuracy:.4f}')
    return np.mean(overall_loss)


def train_single_models(f_model: Type[nn.Module], n: int, transform: Callable,
                        batch_size: int = 256, learning_rate: float = 0.001, epochs: int = 10,
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
            if early_stopping_counter > 20:
                print('EARLY STOPPING, 20 epochs without improving')
                break
        else:
            best_overall_average_loss = overall_average_loss
            best_models = models.copy()
            print('MODELS SAVED')

    # Save models
    for i, model in enumerate(best_models):
        torch.save(model.state_dict(), f'model_{i}.pt')

    # Evaluate on test set
    evaluate(best_models, test_loader, criterion, 0, 0, 'Test', device)


"""
    ###########################
    ENSEMBLE FUNCTIONS
    ###########################
"""
def collect_and_analyze_ensemble_outputs(models, knn, loader, k, pairs, mode='training'):
    """
    Perform KNORA ELIMINATE dynamic ensemble using kNN trained on the training set. The function also
    save the output of the ensemble (logits) together with the input in order to train and
    evaluate the G function
    :param models: pool of models for the dynamic ensemble
    :param knn:
    :param loader:
    :param k:  number of k for NN
    :param pairs:
    :param mode: string referring to training, validation or testing
    :return:
    """
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
        print(f'\t- {mode} accuracy: {corrects / (len(loader) * 16)}')  # 16 is the batch size
    else:
        print(f'\t- {mode} accuracy: {corrects/len(loader)}')


def dynamic_ensemble_cifar(n: int, transform: Callable, k: int):
    """
    Performs KNORA ELIMINATE calling the 'collect_and_analyze_ensemble_outputs' function for each
    set. The function also train kNN using sklearn, storing it after fitting.
    :param n: number of F models used
    :param transform: torchvision transform
    :param k: k for Nearest Neighbor
    :return:
    """
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


"""
    GENERATE MORE DATA
        In order to learn better the distribution of the dynamic ensemble
        created with F networks, I created two functions:
            1. generate_random_rgb_image: generates random rgb images and perform 
                                          the same torchvision transformation to
                                          have additional data for training G
            2. generated_trained_samples: performs KNORA ELIMINATE to each generated
                                          image in order to store the logit used for 
                                          training G
"""
def generate_random_rgb_image(n, transform: Callable, width: int = 32, height: int = 32):
    if not os.path.isdir('generated'):
        os.mkdir('generated')

    for _ in range(n):
        # Generate random pixel values for each channel (R, G, B)
        random_image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

        # Stack the channels to form the RGB image
        transformed_img = transform(random_image)
        numpy_array = np.array(transformed_img)
        np.save(f'generated/{str(uuid.uuid4())[:16]}.npy', numpy_array)

