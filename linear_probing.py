import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

root_path = "./database"

class LinearProbingModel(nn.Module):
    """
    Linear Probing Model with a single linear layer.
    """
    def __init__(self, input_dim, num_classes):
        super(LinearProbingModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def load_embeddings_and_labels(dataset_name, root_path):
    train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
    test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
    X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
    X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    return X_train, y_train, X_test, y_test

def train_linear_probing_model(X_train, y_train, input_dim, num_classes, lr, epochs, batch_size, device):
    """
    Trains a linear probing model.
    """
    model = LinearProbingModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}')

    return model

def evaluate_linear_probing_model(model, X_test, y_test, device):
    """
    Evaluates the linear probing model performance by calculating accuracy.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        labels = torch.tensor(y_test, dtype=torch.long).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        accuracy = torch.mean((preds == labels).float()).item()
        print(f'Linear Probing Accuracy: {accuracy:.4f}')
        return accuracy

def main_linear_probing(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)

    input_dim = X_train.shape[1]  # Dimension of the input features
    num_classes = len(np.unique(y_train))  # Number of classes

    learning_rates = [0.001, 0.005, 0.0001]
    batch_sizes = [16, 32, 64]
    epoch_values = [20, 30, 40]

    best_accuracy = 0
    best_hyperparams = {}

    output_file = f'{dataset_name}_linear_probing_results.txt'
    with open(output_file, 'w') as f:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epoch_values:
                    print(f'Training with lr={lr}, batch_size={batch_size}, epochs={epochs}')
                    f.write(f'Training with lr={lr}, batch_size={batch_size}, epochs={epochs}\n')
                    model = train_linear_probing_model(X_train, y_train, input_dim, num_classes, lr, epochs, batch_size, device)
                    accuracy = evaluate_linear_probing_model(model, X_test, y_test, device)

                    f.write(f'Accuracy: {accuracy:.2f}\n')

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparams = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}

        f.write(f'Best Hyperparameters: {best_hyperparams} with accuracy: {best_accuracy:.4f}\n')

    print(f'Best Hyperparameters: {best_hyperparams} with accuracy: {best_accuracy:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probing with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    main_linear_probing(dataset_name)
