import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

root_path = "./database"

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

def load_embeddings_and_labels(dataset_name, root_path):
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
    knn = CustomKNN(k=k, device=device)
    knn.fit(X_train, y_train)
    return knn

def predict_with_knn(knn, X):
    return knn.predict(X)

def evaluate_knn_performance(y_pred, y_true):
    accuracy = np.mean(y_pred == y_true)
    return accuracy

def main_knn(dataset_name):
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 40]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_file = f'{dataset_name}_knn_results.txt'

    best_k = None
    best_val_accuracy = 0
    val_accuracies = {}

    print(f'\nEvaluating {len(k_values)} values of k:')
    with open(output_file, 'w') as f:
        f.write(f'\nEvaluating {len(k_values)} values of k:\n')
        for k in k_values:
            knn = fit_knn_model(X_train, y_train, k, device)
            y_val_pred = predict_with_knn(knn, X_val)
            val_accuracy = evaluate_knn_performance(y_val_pred, y_val)
            val_accuracies[k] = val_accuracy
            print(f'k={k}: Validation Accuracy = {val_accuracy:.4f}')
            f.write(f'k={k}: Validation Accuracy = {val_accuracy:.4f}\n')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_k = k

    print(f'\nBest k: {best_k} with validation accuracy: {best_val_accuracy:.4f}')
    with open(output_file, 'a') as f:
        f.write(f'\nBest k: {best_k} with validation accuracy: {best_val_accuracy:.4f}\n')

    knn = fit_knn_model(X_train, y_train, best_k, device)
    y_test_pred = predict_with_knn(knn, X_test)
    test_accuracy = evaluate_knn_performance(y_test_pred, y_test)

    print(f'Final kNN Accuracy on test set: {test_accuracy:.4f}')
    with open(output_file, 'a') as f:
        f.write(f'Final kNN Accuracy on test set: {test_accuracy:.4f}\n')

    misclassified_indices = np.where(y_test_pred != y_test)[0]
    if len(misclassified_indices) > 0:
        print(f'Misclassified Indices: {misclassified_indices[:10]}')
        print(f'Total Misclassified: {len(misclassified_indices)}')
        with open(output_file, 'a') as f:
            f.write(f'Misclassified Indices: {misclassified_indices[:10]}\n')
            f.write(f'Total Misclassified: {len(misclassified_indices)}\n')
    else:
        print("No misclassifications found.")
        with open(output_file, 'a') as f:
            f.write("No misclassifications found.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    main_knn(dataset_name)
