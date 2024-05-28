import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import KFold
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
    train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
    test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
    X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
    X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    return X_train, y_train, X_test, y_test

def fit_knn_model(X_train, y_train, k, device):
    knn = CustomKNN(k=k, device=device)
    knn.fit(X_train, y_train)
    return knn

def predict_with_knn(knn, X_test):
    return knn.predict(X_test)

def evaluate_knn_performance(y_pred, y_test):
    accuracy = np.mean(y_pred == y_test)
    print(f'kNN Accuracy: {accuracy:.2f}')
    return accuracy

def cross_validate_knn(X_train, y_train, k_values, num_folds, device, output_file):
    results = {}
    print(f'\nEvaluating {num_folds} folds:')
    with open(output_file, 'w') as f:
        f.write(f'\nEvaluating {num_folds} folds:\n')
        accuracies = {k: [] for k in k_values}
        kf = KFold(n_splits=num_folds)

        for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            for k in k_values:
                knn = CustomKNN(k=k, device=device)
                knn.fit(X_train_fold, y_train_fold)
                y_pred = knn.predict(X_val_fold)
                accuracy = np.mean(y_pred == y_val_fold)
                accuracies[k].append(accuracy)
                print(f'Fold {fold}, k={k}: Accuracy = {accuracy:.4f}')
                f.write(f'Fold {fold}, k={k}: Accuracy = {accuracy:.4f}\n')

        avg_accuracies = {k: np.mean(acc) for k, acc in accuracies.items()}
        best_k = max(avg_accuracies, key=avg_accuracies.get)
        results[num_folds] = avg_accuracies

        print("\nAverage Accuracies for {} folds:".format(num_folds))
        f.write("\nAverage Accuracies for {} folds:\n".format(num_folds))
        for k, avg_accuracy in avg_accuracies.items():
            print(f'k={k}: Average Accuracy = {avg_accuracy:.4f}')
            f.write(f'k={k}: Average Accuracy = {avg_accuracy:.4f}\n')

        print("\nSummary for {} folds:\n".format(num_folds))
        f.write("\nSummary for {} folds:\n".format(num_folds))
        for k in k_values:
            best_indicator = " Best" if k == best_k else ""
            print(f'K{k} = {avg_accuracies[k]:.4f}{best_indicator}')
            f.write(f'K{k} = {avg_accuracies[k]:.4f}{best_indicator}\n')

    return results

def main_knn(dataset_name):
    X_train, y_train, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 20, 30, 40]
    num_folds = 15

    output_file = f'{dataset_name}_knn_results.txt'
    results = cross_validate_knn(X_train, y_train, k_values, num_folds, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), output_file)

    best_k_overall = None
    best_accuracy = 0

    for num_folds, accuracies in results.items():
        best_k = max(accuracies, key=accuracies.get)
        if accuracies[best_k] > best_accuracy:
            best_accuracy = accuracies[best_k]
            best_k_overall = best_k

    print(f'\nBest k: {best_k_overall} with accuracy: {best_accuracy:.4f}')

    knn = fit_knn_model(X_train, y_train, best_k_overall, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y_pred = predict_with_knn(knn, X_test)
    accuracy = evaluate_knn_performance(y_pred, y_test)

    with open(output_file, 'a') as f:
        f.write(f'\nBest k: {best_k_overall} with accuracy: {best_accuracy:.4f}\n')
        f.write(f'Final kNN Accuracy on test set: {accuracy:.4f}\n')

    misclassified_indices = np.where(y_pred != y_test)[0]
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
