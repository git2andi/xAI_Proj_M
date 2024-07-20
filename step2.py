import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import logging

root_path = "./database"

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomKNN:
    """
    Custom k-Nearest Neighbors (kNN) classifier with GPU support.
    """
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y):
        """
        Fits the kNN model using the training data.
        """
        self.X_train = torch.tensor(X, device=self.device)
        self.y_train = torch.tensor(y, device=self.device)

    def predict(self, X):
        """
        Predicts the labels for the input data using kNN.
        """
        X = torch.tensor(X, device=self.device)
        distances = torch.cdist(X, self.X_train)
        neighbors = distances.argsort(dim=1)[:, :self.k]
        top_labels = self.y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

def load_embeddings_and_labels(dataset_name, root_path):
    """
    Load embeddings and labels for the specified dataset.
    """
    if dataset_name in ["dermamnist", "breastmnist"]:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
        val_data = np.load(os.path.join(root_path, dataset_name, 'val.npz'))
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_val, y_val = val_data['embeddings'], val_data['labels'].reshape(-1,)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    else:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def fit_knn_model(X_train, y_train, k, device):
    """
    Train the kNN model with the specified number of neighbors (k).
    """
    knn = CustomKNN(k=k, device=device)
    knn.fit(X_train, y_train)
    return knn

def predict_with_knn(knn, X):
    """
    Make predictions using the kNN model.
    """
    return knn.predict(X)

def evaluate_knn_performance(y_pred, y_true):
    """
    Evaluate the performance of the kNN model using various metrics.
    """
    accuracy = accuracy_score(y_pred, y_true)
    precision = precision_score(y_pred, y_true, average='weighted')
    recall = recall_score(y_pred, y_true, average='weighted')
    f1 = f1_score(y_pred, y_true, average='weighted')
    return accuracy, precision, recall, f1

def train_linear_probe(X_train, y_train, X_val, y_val, input_dim, num_classes, hyperparams, device):
    """
    Train a linear probe on the embeddings with specified hyperparameters.
    """
    model = LinearProbe(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

    best_val_accuracy = 0
    for epoch in range(hyperparams['epochs']):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_labels = torch.tensor(y_val, dtype=torch.long).to(device)
            val_outputs = model(val_inputs)
            _, preds = torch.max(val_outputs, 1)
            val_accuracy = accuracy_score(val_labels.cpu(), preds.cpu())
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

    return model, best_val_accuracy

def evaluate_linear_probe(model, X_test, y_test, device):
    """
    Evaluate the performance of the linear probe on the test set.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        labels = torch.tensor(y_test, dtype=torch.long).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_accuracy = accuracy_score(labels.cpu(), preds.cpu())
        test_precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
        test_recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
        test_f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        misclassified_indices = np.where(preds.cpu().numpy() != y_test)[0]
    return test_accuracy, test_precision, test_recall, test_f1, misclassified_indices

def main(dataset_name):
    # Hyperparameters for linear probing
    lrs = [0.001, 0.005]
    batch_sizes = [32, 64]
    epochs_list = [20, 40]

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluate kNN
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 40]
    output_file = f'{dataset_name}_results.txt'

    best_k = None
    best_val_accuracy = 0
    val_accuracies = {}

    logging.info(f'Evaluating {len(k_values)} values of k:')
    with open(output_file, 'w') as f:
        f.write(f'\nEvaluating {len(k_values)} values of k:\n')
        for k in k_values:
            knn = fit_knn_model(X_train, y_train, k, device)
            y_val_pred = predict_with_knn(knn, X_val)
            val_accuracy, _, _, _ = evaluate_knn_performance(y_val_pred, y_val)
            val_accuracies[k] = val_accuracy
            logging.info(f'k={k}: Validation Accuracy = {val_accuracy:.4f}')
            f.write(f'k={k}: Validation Accuracy = {val_accuracy:.4f}\n')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_k = k

        logging.info(f'Best k: {best_k} with validation accuracy: {best_val_accuracy:.4f}')
        f.write(f'\nBest k: {best_k} with validation accuracy: {best_val_accuracy:.4f}\n')

        knn = fit_knn_model(X_train, y_train, best_k, device)
        y_test_pred = predict_with_knn(knn, X_test)
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_knn_performance(y_test_pred, y_test)

        logging.info(f'Final kNN Accuracy on test set: {test_accuracy:.4f}')
        logging.info(f'Final kNN Precision on test set: {test_precision:.4f}')
        logging.info(f'Final kNN Recall on test set: {test_recall:.4f}')
        logging.info(f'Final kNN F1 Score on test set: {test_f1:.4f}')
        f.write(f'Final kNN Accuracy on test set: {test_accuracy:.4f}\n')
        f.write(f'Final kNN Precision on test set: {test_precision:.4f}\n')
        f.write(f'Final kNN Recall on test set: {test_recall:.4f}\n')
        f.write(f'Final kNN F1 Score on test set: {test_f1:.4f}\n')

        misclassified_indices = np.where(y_test_pred != y_test)[0]
        if len(misclassified_indices) > 0:
            logging.info(f'Misclassified Indices: {misclassified_indices[:10]}')
            logging.info(f'Total Misclassified: {len(misclassified_indices)}')
            f.write(f'Misclassified Indices: {misclassified_indices[:10]}\n')
            f.write(f'Total Misclassified: {len(misclassified_indices)}\n')
        else:
            logging.info("No misclassifications found.")
            f.write("No misclassifications found.\n")

    # Evaluate Linear Probing
    best_hyperparams = None
    best_val_accuracy = 0

    logging.info(f'Training linear probe for {dataset_name}')
    with open(output_file, 'a') as f:
        for lr in lrs:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    hyperparams = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
                    model, val_accuracy = train_linear_probe(X_train, y_train, X_val, y_val, input_dim, num_classes, hyperparams, device)
                    logging.info(f'Validation accuracy with hyperparams {hyperparams}: {val_accuracy:.4f}')
                    f.write(f'Validation accuracy with hyperparams {hyperparams}: {val_accuracy:.4f}\n')
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_hyperparams = hyperparams

        logging.info(f'Best hyperparameters: {best_hyperparams} with validation accuracy: {best_val_accuracy:.4f}')
        f.write(f'\nBest Hyperparameters: {best_hyperparams} with validation accuracy: {best_val_accuracy:.4f}\n')

        # Evaluate the best model on the test set
        model, _ = train_linear_probe(X_train, y_train, X_val, y_val, input_dim, num_classes, best_hyperparams, device)
        test_accuracy, test_precision, test_recall, test_f1, misclassified_indices = evaluate_linear_probe(model, X_test, y_test, device)
        logging.info(f'Linear probe test accuracy: {test_accuracy:.4f}')
        logging.info(f'Linear probe test precision: {test_precision:.4f}')
        logging.info(f'Linear probe test recall: {test_recall:.4f}')
        logging.info(f'Linear probe test F1 score: {test_f1:.4f}')
        logging.info(f'Total misclassified: {len(misclassified_indices)}')
        if len(misclassified_indices) > 0:
            logging.info(f'Misclassified Indices: {misclassified_indices[:10]}')
        
        f.write(f'Test accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test precision: {test_precision:.4f}\n')
        f.write(f'Test recall: {test_recall:.4f}\n')
        f.write(f'Test F1 score: {test_f1:.4f}\n')
        f.write(f'Total misclassified: {len(misclassified_indices)}\n')
        if len(misclassified_indices) > 0:
            f.write(f'Misclassified Indices: {misclassified_indices[:10]}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN and Linear Probing with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    main(dataset_name)
