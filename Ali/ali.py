import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
import time
from bayes_opt import BayesianOptimization

batch_size=200
root_path = "./database"

def load_embeddings_and_labels(dataset_name, root_path):
    if dataset_name in ["dermamnist", "breastmnist"]:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'), mmap_mode='r')
        val_data = np.load(os.path.join(root_path, dataset_name, 'val.npz'), mmap_mode='r')
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'), mmap_mode='r')
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_val, y_val = val_data['embeddings'], val_data['labels'].reshape(-1,)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    else:
        train_data = np.load(os.path.join(root_path, dataset_name, 'train.npz'), mmap_mode='r')
        test_data = np.load(os.path.join(root_path, dataset_name, 'test.npz'), mmap_mode='r')
        X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)

    print(f"Loaded dataset {dataset_name} with shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def stratified_bootstrap_sample(X, y):
    unique_classes = np.unique(y)
    X_resampled, y_resampled = [], []
    for cls in unique_classes:
        X_class, y_class = X[y == cls], y[y == cls]
        X_class_resampled, y_class_resampled = resample(X_class, y_class)
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)
    return np.vstack(X_resampled), np.hstack(y_resampled)

def balanced_bootstrap_sample(X, y):
    unique_classes = np.unique(y)
    max_class_size = max([np.sum(y == cls) for cls in unique_classes])
    X_resampled, y_resampled = [], []
    for cls in unique_classes:
        X_class, y_class = X[y == cls], y[y == cls]
        X_class_resampled, y_class_resampled = resample(X_class, y_class, replace=True, n_samples=max_class_size)
        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)
    return np.vstack(X_resampled), np.hstack(y_resampled)

def out_of_bag_bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    return X[indices], y[indices], X[oob_indices], y[oob_indices]

def smote_bootstrap_sample(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return resample(X_resampled, y_resampled)

def apply_projection(X, method='random', n_components=None, variance_threshold=None):
    if method == 'random':
        if n_components is None:
            raise ValueError("n_components must be specified for random projection")
        transformer = GaussianRandomProjection(n_components=n_components)
    elif method == 'pca':
        if variance_threshold is not None:
            transformer = PCA(n_components=variance_threshold)
        elif n_components is not None:
            transformer = PCA(n_components=n_components)
        else:
            raise ValueError("Either n_components or variance_threshold must be specified for PCA")
    X_projected = transformer.fit_transform(X)
    return X_projected, transformer

def apply_projection_to_test(X_test, transformer):
    return transformer.transform(X_test)

class CustomKNN:
    def __init__(self, k=5, distance_metric='euclidean', device='cuda'):
        self.k = k
        self.distance_metric = distance_metric
        self.device = device

    def fit(self, X, y):
        self.X_train = torch.tensor(X, device=self.device)
        self.y_train = torch.tensor(y, device=self.device)

    def predict(self, X, batch_size=batch_size):
        X = torch.tensor(X, device=self.device)
        num_samples = X.shape[0]
        predictions = []
        total_time = 0

        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            start_time = time.time()
            
            if self.distance_metric == 'euclidean':
                distances = torch.cdist(batch, self.X_train)
            elif self.distance_metric == 'manhattan':
                distances = torch.cdist(batch, self.X_train, p=1)
            elif self.distance_metric == 'cosine':
                distances = 1 - F.cosine_similarity(batch.unsqueeze(1), self.X_train.unsqueeze(0), dim=2)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            end_time = time.time()
            total_time += (end_time - start_time)
            
            neighbors = distances.argsort(dim=1)[:, :self.k]
            top_labels = self.y_train[neighbors]
            batch_predictions = torch.mode(top_labels, dim=1).values
            predictions.append(batch_predictions.cpu().numpy())
        
        print(f"Total time for {self.distance_metric} distance: {total_time:.3f} seconds")
        return np.concatenate(predictions, axis=0)

def train_knn_model(X, y, k, distance_metric='euclidean', device='cuda'):
    knn = CustomKNN(k=k, distance_metric=distance_metric, device=device)
    start_time = time.time()
    knn.fit(X, y)
    training_time = time.time() - start_time
    return knn, training_time

def predict_with_ensemble(classifiers, X_test_samples, batch_size=batch_size):
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test, batch_size=batch_size)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions

def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, variance_threshold, k, n_classifiers, distance_metric, sampling_method, output_file, training_times):
    y_pred = predict_with_ensemble(classifiers, X_test_samples)
    accuracy = accuracy_score(y_test, y_pred)
    total_training_time = sum(training_times)
    
    print(f"Accuracy for method={method}, n_components={n_components}, variance_threshold={variance_threshold}, k={k}, n_classifiers={n_classifiers}, distance_metric={distance_metric}, sampling_method={sampling_method}: {accuracy:.4f}")
    print(f"Total training time: {total_training_time:.3f} seconds")

    results = {
        'method': method,
        'n_components': n_components,
        'variance_threshold': variance_threshold,
        'k': k,
        'n_classifiers': n_classifiers,
        'distance_metric': distance_metric,
        'sampling_method': sampling_method,
        'accuracy': accuracy,
        'training_time_seconds': round(total_training_time, 3)
    }

    print(f"Results: {results}")
    
    df = pd.DataFrame([results])
    print(f"DataFrame to be saved:\n{df}")

    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Results saved to {output_file}")

def bayesian_optimization():
    param_space = {
        'method': [0, 1],  # 0 for 'random', 1 for 'pca'
        'n_components': [50, 100, 150],
        'variance_threshold': [0.9, 0.95, 0.99],
        'k': [3, 5, 10],
        'n_classifiers': [10, 20, 30],
        'distance_metric': [0, 1, 2],  # 0 for 'euclidean', 1 for 'manhattan', 2 for 'cosine'
        'sampling_method': [0, 1, 2, 3]  # 0 for 'stratified', 1 for 'balanced', 2 for 'oob', 3 for 'smote'
    }

    def evaluate_model(method, n_components, variance_threshold, k, n_classifiers, distance_metric, sampling_method):
        method = ['random', 'pca'][int(method)]
        distance_metric = ['euclidean', 'manhattan', 'cosine'][int(distance_metric)]
        sampling_method = ['stratified', 'balanced', 'oob', 'smote'][int(sampling_method)]
        
        classifiers = []
        X_val_samples = []
        X_test_samples = []
        training_times = []

        for i in range(int(n_classifiers)):
            if sampling_method == 'stratified':
                X_sample, y_sample = stratified_bootstrap_sample(X_train, y_train)
            elif sampling_method == 'balanced':
                X_sample, y_sample = balanced_bootstrap_sample(X_train, y_train)
            elif sampling_method == 'oob':
                X_sample, y_sample, _, _ = out_of_bag_bootstrap_sample(X_train, y_train)
            elif sampling_method == 'smote':
                X_sample, y_sample = smote_bootstrap_sample(X_train, y_train)
            else:
                raise ValueError(f"Unsupported sampling method: {sampling_method}")

            X_projected, transformer = apply_projection(X_sample, method=method, n_components=int(n_components) if method == 'random' else None, variance_threshold=float(variance_threshold) if method == 'pca' else None)
            X_val_projected = apply_projection_to_test(X_val, transformer)
            X_test_projected = apply_projection_to_test(X_test, transformer)

            knn, training_time = train_knn_model(X_projected, y_sample, k=int(k), distance_metric=distance_metric, device='cuda')
            classifiers.append(knn)
            X_val_samples.append(X_val_projected)
            X_test_samples.append(X_test_projected)
            training_times.append(training_time)

        y_pred = predict_with_ensemble(classifiers, X_test_samples)
        accuracy = accuracy_score(y_test, y_pred)
        total_training_time = sum(training_times)

        return accuracy

    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds={
            'method': (0, 1),
            'n_components': (50, 150),
            'variance_threshold': (0.9, 0.99),
            'k': (3, 10),
            'n_classifiers': (10, 30),
            'distance_metric': (0, 2),
            'sampling_method': (0, 3)
        },
        random_state=42,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=50,
    )

    best_params = optimizer.max['params']
    best_accuracy = optimizer.max['target']

    best_params['method'] = ['random', 'pca'][int(best_params['method'])]
    best_params['distance_metric'] = ['euclidean', 'manhattan', 'cosine'][int(best_params['distance_metric'])]
    best_params['sampling_method'] = ['stratified', 'balanced', 'oob', 'smote'][int(best_params['sampling_method'])]

    total_training_time = sum([res['target'][1] for res in optimizer.res if res['params'] == best_params])

    output_file = f'{dataset_name}_forest_knn_results.csv'

    results = {
        'method': best_params['method'],
        'n_components': int(best_params['n_components']),
        'variance_threshold': best_params['variance_threshold'],
        'k': int(best_params['k']),
        'n_classifiers': int(best_params['n_classifiers']),
        'distance_metric': best_params['distance_metric'],
        'sampling_method': best_params['sampling_method'],
        'accuracy': best_accuracy,
        'training_time_seconds': round(total_training_time, 3)
    }

    df = pd.DataFrame([results])
    print(f"DataFrame to be saved:\n{df}")

    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Starting Forest-kNN on dataset {dataset_name}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    bayesian_optimization()
    print("Forest-kNN evaluation completed.")
