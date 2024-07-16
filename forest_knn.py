import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import pandas as pd
import datetime
from joblib import Parallel, delayed

root_path = "./database"

def load_embeddings_and_labels(dataset_name, root_path, subsample=True):
    data = np.load(os.path.join(root_path, dataset_name, 'train.npz'))
    X, y = data['embeddings'], data['labels'].reshape(-1,)
    if subsample:
        X, y = X[::10], y[::10]
    return X, y

def load_full_embeddings_and_labels(dataset_name, root_path):
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

def create_random_subsets(X, y, n_subsets, subset_size, feature_fraction):
    subsets = []
    n_samples, n_features = X.shape
    subset_feature_size = int(n_features * feature_fraction)
    
    for _ in range(n_subsets):
        sample_indices = np.random.choice(n_samples, subset_size, replace=True)
        feature_indices = np.random.choice(n_features, subset_feature_size, replace=True)
        X_subset = X[sample_indices][:, feature_indices]
        y_subset = y[sample_indices]
        subsets.append((X_subset, y_subset, feature_indices))
    
    return subsets

class CustomKNN:
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y, feature_indices):
        self.X_train = torch.tensor(X, device=self.device)
        self.y_train = torch.tensor(y, device=self.device)
        self.feature_indices = feature_indices

    def predict(self, X):
        X = torch.tensor(X[:, self.feature_indices], device=self.device)
        distances = torch.cdist(X, self.X_train)
        neighbors = distances.argsort(dim=1)[:, :self.k]
        top_labels = self.y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

class ForestKNN:
    def __init__(self, k=5, n_estimators=10, subset_size=100, feature_fraction=0.8, device='cuda'):
        self.k = k
        self.n_estimators = n_estimators
        self.subset_size = subset_size
        self.feature_fraction = feature_fraction
        self.device = device
        self.classifiers = []

    def fit(self, X, y):
        subsets = create_random_subsets(X, y, self.n_estimators, self.subset_size, self.feature_fraction)
        for X_subset, y_subset, feature_indices in subsets:
            knn = CustomKNN(k=self.k, device=self.device)
            knn.fit(X_subset, y_subset, feature_indices)
            self.classifiers.append(knn)

    def predict(self, X):
        all_predictions = Parallel(n_jobs=-1)(delayed(knn.predict)(X) for knn in self.classifiers)
        all_predictions = np.array(all_predictions).T
        majority_votes = mode(all_predictions, axis=1)[0].flatten()
        return majority_votes

def evaluate_knn_performance(y_pred, y_true):
    accuracy = np.mean(y_pred == y_true)
    return accuracy

def save_results_to_csv(dataset_name, results, filename="results.csv"):
    columns = ["dataset", "model_type", "k", "n_estimators", "subset_size", "feature_fraction", "accuracy", "timestamp"]
    df = pd.DataFrame(results, columns=columns)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def stability_check(accuracies, threshold=0.01, min_runs=5):
    if len(accuracies) < min_runs:
        return False
    recent_accuracies = accuracies[-min_runs:]
    variation = np.std(recent_accuracies) / np.mean(recent_accuracies)
    return variation < threshold


def adaptive_hyperparameter_tuning(dataset_name, subsample=True):
    if subsample:
        X_train, y_train = load_embeddings_and_labels(dataset_name, root_path, subsample)
        # Load the test set
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'))
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = load_full_embeddings_and_labels(dataset_name, root_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []

    # Train and evaluate baseline k-NN
    baseline_knn = CustomKNN(k=5, device=device)
    baseline_knn.fit(X_train, y_train, np.arange(X_train.shape[1]))
    y_test_pred_baseline = baseline_knn.predict(X_test)
    test_accuracy_baseline = evaluate_knn_performance(y_test_pred_baseline, y_test)
    print(f'Baseline k-NN Test Accuracy: {test_accuracy_baseline:.4f}')

    results.append([
        dataset_name,
        "Baseline k-NN",
        5,  # k
        None,  # n_estimators
        None,  # subset_size
        None,  # feature_fraction
        test_accuracy_baseline,
        datetime.datetime.now()
    ])

    def stability_check(accuracies, threshold=0.01, min_runs=5):
        if len(accuracies) < min_runs:
            return False
        recent_accuracies = accuracies[-min_runs:]
        variation = np.std(recent_accuracies) / np.mean(recent_accuracies)
        return variation < threshold

    def iterative_tuning(parameter_name, initial_value, increment, model_func, results, dataset_name, **model_params):
        value = initial_value
        stable = False
        best_value = value
        best_accuracy = 0
        no_improvement_runs = 0
        max_no_improvement_runs = 3  # Stop after 3 runs without improvement

        while not stable and no_improvement_runs < max_no_improvement_runs:
            accuracies = []
            for _ in range(5):
                model = model_func(**model_params, **{parameter_name: value})
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_accuracy = evaluate_knn_performance(y_test_pred, y_test)
                accuracies.append(test_accuracy)
                print(f'Forest k-NN ({parameter_name}={value}) Test Accuracy: {test_accuracy:.4f}')

                results.append([
                    dataset_name,
                    "Forest k-NN",
                    model_params.get('k', None),
                    model_params.get('n_estimators', None),
                    model_params.get('subset_size', None),
                    model_params.get('feature_fraction', None),
                    test_accuracy,
                    datetime.datetime.now()
                ])

            current_best_accuracy = max(accuracies)
            if current_best_accuracy > best_accuracy:
                best_value = value
                best_accuracy = current_best_accuracy
                no_improvement_runs = 0
            else:
                no_improvement_runs += 1
            
            if stability_check(accuracies):
                stable = True
            else:
                value += increment
            
        return best_value, best_accuracy

    # Initial values
    initial_k = 3
    initial_n_estimators = 5
    initial_subset_size = 50
    initial_feature_fraction = 0.5

    # Tune k
    best_k, _ = iterative_tuning(
        parameter_name='k',
        initial_value=initial_k,
        increment=2,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        n_estimators=initial_n_estimators,
        subset_size=initial_subset_size,
        feature_fraction=initial_feature_fraction,
        device=device
    )

    # Tune n_estimators
    best_n_estimators, _ = iterative_tuning(
        parameter_name='n_estimators',
        initial_value=initial_n_estimators,
        increment=5,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        subset_size=initial_subset_size,
        feature_fraction=initial_feature_fraction,
        device=device
    )

    # Tune subset_size
    best_subset_size, _ = iterative_tuning(
        parameter_name='subset_size',
        initial_value=initial_subset_size,
        increment=50,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        n_estimators=best_n_estimators,
        feature_fraction=initial_feature_fraction,
        device=device
    )

    # Tune feature_fraction
    best_feature_fraction, best_accuracy = iterative_tuning(
        parameter_name='feature_fraction',
        initial_value=initial_feature_fraction,
        increment=0.1,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        n_estimators=best_n_estimators,
        subset_size=best_subset_size,
        device=device
    )

    # Recheck previous hyperparameters for further improvement
    best_k, _ = iterative_tuning(
        parameter_name='k',
        initial_value=best_k,
        increment=2,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        n_estimators=best_n_estimators,
        subset_size=best_subset_size,
        feature_fraction=best_feature_fraction,
        device=device
    )

    best_n_estimators, _ = iterative_tuning(
        parameter_name='n_estimators',
        initial_value=best_n_estimators,
        increment=5,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        subset_size=best_subset_size,
        feature_fraction=best_feature_fraction,
        device=device
    )

    best_subset_size, _ = iterative_tuning(
        parameter_name='subset_size',
        initial_value=best_subset_size,
        increment=50,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        n_estimators=best_n_estimators,
        feature_fraction=best_feature_fraction,
        device=device
    )

    best_feature_fraction, best_accuracy = iterative_tuning(
        parameter_name='feature_fraction',
        initial_value=best_feature_fraction,
        increment=0.1,
        model_func=ForestKNN,
        results=results,
        dataset_name=dataset_name,
        k=best_k,
        n_estimators=best_n_estimators,
        subset_size=best_subset_size,
        device=device
    )

    print(f'Best Forest k-NN Parameters: k={best_k}, n_estimators={best_n_estimators}, subset_size={best_subset_size}, feature_fraction={best_feature_fraction}')
    print(f'Best Forest k-NN Test Accuracy: {best_accuracy:.4f}')

    save_results_to_csv(dataset_name, results)

    return test_accuracy_baseline, best_accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forest k-NN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    parser.add_argument("--subsampled", action="store_true", help="Use subsampled dataset")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    args = parser.parse_args()

    dataset_name = args.dataset
    subsample = args.subsampled

    adaptive_hyperparameter_tuning(dataset_name, subsample)
