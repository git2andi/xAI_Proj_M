import os
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import logging
import random
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import Isomap

root_path = "./database"

def setup_logging(dataset_name):
    logger_name = f'forest_knn_logger_{dataset_name}'
    logger = logging.getLogger(logger_name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(f'{dataset_name}_forest_knn.log')
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

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
    
    logger.info(f"Loaded dataset {dataset_name} with shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_bootstrap_sample(X, y, method='bootstrap', random_state=42):
    if method == 'bootstrap':
        return resample(X, y, random_state=random_state)
    elif method == 'stratified':
        unique_classes = np.unique(y)
        X_resampled, y_resampled = [], []
        for cls in unique_classes:
            X_class, y_class = X[y == cls], y[y == cls]
            X_class_resampled, y_class_resampled = resample(X_class, y_class, random_state=random_state)
            X_resampled.append(X_class_resampled)
            y_resampled.append(y_class_resampled)
        return np.vstack(X_resampled), np.hstack(y_resampled)
    elif method == 'balanced':
        unique_classes = np.unique(y)
        max_class_size = max([np.sum(y == cls) for cls in unique_classes])
        X_resampled, y_resampled = [], []
        for cls in unique_classes:
            X_class, y_class = X[y == cls], y[y == cls]
            X_class_resampled, y_class_resampled = resample(X_class, y_class, replace=True, n_samples=max_class_size, random_state=random_state)
            X_resampled.append(X_class_resampled)
            y_resampled.append(y_class_resampled)
        return np.vstack(X_resampled), np.hstack(y_resampled)
    elif method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    elif method == 'oob':
        n_samples = len(X)
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        return X[indices], y[indices], X[oob_indices], y[oob_indices]
    else:
        raise ValueError("Unknown sampling method.")

def apply_projection(X, y=None, method='random', n_components=None, variance_threshold=None, random_state=42):
    if n_components is not None:
        n_components = int(n_components)
    if method == 'random':
        if n_components is None:
            raise ValueError("n_components must be specified for random projection")
        transformer = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    elif method == 'pca':
        if variance_threshold is not None:
            transformer = PCA(n_components=variance_threshold, random_state=random_state)
        elif n_components is not None:
            transformer = PCA(n_components=n_components, random_state=random_state)
        else:
            raise ValueError("Either n_components or variance_threshold must be specified for PCA")
    elif method == 'isomap':
        transformer = Isomap(n_components=n_components)
    else:
        raise ValueError("Unknown projection method.")
    
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

    def predict(self, X, batch_size=200):
        X = torch.tensor(X, device=self.device)
        num_samples = X.shape[0]
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            if self.distance_metric == 'euclidean':
                distances = torch.cdist(batch, self.X_train)
            elif self.distance_metric == 'manhattan':
                distances = torch.cdist(batch, self.X_train, p=1)
            elif self.distance_metric == 'cosine':
                distances = 1 - F.cosine_similarity(batch.unsqueeze(1), self.X_train.unsqueeze(0), dim=2)
            elif self.distance_metric == 'minkowski':
                distances = torch.cdist(batch, self.X_train, p=2)
            elif self.distance_metric == 'chebyshev':
                distances = torch.cdist(batch, self.X_train, p=float('inf'))
            elif self.distance_metric == 'hamming':
                distances = torch.cdist(batch, self.X_train, p=0)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
            neighbors = distances.argsort(dim=1)[:, :self.k]
            top_labels = self.y_train[neighbors]
            batch_predictions = torch.mode(top_labels, dim=1).values
            predictions.append(batch_predictions.cpu().numpy())
            torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
        
        return np.concatenate(predictions, axis=0)

def bayesian_optimization_knn(X_train, y_train):
    def knn_evaluate(method, n_components, variance_threshold, n_neighbors, weights_idx, metric_idx, n_classifiers, sampling_method):
        method = ['random', 'pca', 'isomap'][int(method)]
        weights_options = ['uniform', 'distance']
        metric_options = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski', 'hamming']
        weights = weights_options[int(weights_idx)]
        metric = metric_options[int(metric_idx)]
        sampling_method = ['bootstrap', 'stratified', 'balanced', 'smote', 'oob'][int(sampling_method)]
        n_components = int(n_components)

        X_projected, transformer = apply_projection(X_train, method=method, n_components=n_components, variance_threshold=variance_threshold, random_state=42)
    
        knn = CustomKNN(k=int(n_neighbors), distance_metric=metric, device='cuda')
        knn.fit(X_projected, y_train)
        score = knn.predict(X_projected)
        return accuracy_score(y_train, score)

    pbounds = {
        'method': (0, 2),
        'n_components': (50, 150),
        'variance_threshold': (0.9, 0.99),
        'n_neighbors': (3, 10),
        'weights_idx': (0, 1),
        'metric_idx': (0, 5),
        'n_classifiers': (10, 30),
        'sampling_method': (0, 4)
    }

    optimizer = BayesianOptimization(
        f=knn_evaluate,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=10, n_iter=50)

    best_params = optimizer.max['params']
    best_accuracy = optimizer.max['target']
    best_params['method'] = ['random', 'pca', 'isomap'][int(best_params['method'])]
    best_params['weights'] = ['uniform', 'distance'][int(best_params.pop('weights_idx'))]
    best_params['metric'] = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski', 'hamming'][int(best_params.pop('metric_idx'))]
    best_params['sampling_method'] = ['bootstrap', 'stratified', 'balanced', 'smote', 'oob'][int(best_params.pop('sampling_method'))]

    if best_params['method'] == 'pca' and best_params['variance_threshold'] is not None:
        transformer = PCA(n_components=best_params['variance_threshold'], random_state=42)
    elif best_params['method'] in ['random', 'pca'] and best_params['n_components'] is not None:
        transformer = PCA(n_components=int(best_params['n_components']), random_state=42) if best_params['method'] == 'pca' else GaussianRandomProjection(n_components=int(best_params['n_components']), random_state=42)
    elif best_params['method'] == 'isomap':
        transformer = Isomap(n_components=int(best_params['n_components']))
    else:
        raise ValueError("Unknown projection method.")

    transformer.fit(X_train)
    X_projected = transformer.transform(X_train)

    best_knn = CustomKNN(
        k=int(best_params['n_neighbors']),
        distance_metric=best_params['metric'],
        device='cuda'
    )
    best_knn.fit(X_projected, y_train)

    logger.info(f"Best parameters found: {best_params} with accuracy: {best_accuracy:.4f}")

    return best_knn, best_params, transformer

def predict_with_ensemble(classifiers, X_test_samples, batch_size=200):
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test, batch_size=batch_size)
        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions

def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, variance_threshold, n_classifiers, output_file, best_params, sampling_method, X_oob=None, y_oob=None):
    y_pred = predict_with_ensemble(classifiers, X_test_samples)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    k = best_params['n_neighbors']
    
    logger.info(f"Accuracy for method={method}, n_components={n_components}, variance_threshold={variance_threshold}, k={k}, n_classifiers={n_classifiers}, sampling_method={sampling_method}: {accuracy:.4f}")
    
    results = {
        'method': method,
        'n_components': n_components,
        'variance_threshold': variance_threshold,
        'k': k,
        'n_classifiers': n_classifiers,
        'best_params': best_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sampling_method': sampling_method,
        'total_misclassified': len(misclassified_indices),
        'misclassified_indices': random.sample(list(misclassified_indices), 5) if len(misclassified_indices) > 0 else 'None'
    }

    df = pd.DataFrame([results])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    logger.info(f"Results saved to {output_file}")

def main_forest_knn(dataset_name):
    global logger
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Forest-KNN on dataset {dataset_name}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)

    output_file = f'{dataset_name}_forest_knn_results.csv'
    
    best_knn, best_params, transformer = bayesian_optimization_knn(X_train, y_train)

    method = best_params['method']
    n_components = int(best_params['n_components'])
    variance_threshold = best_params.get('variance_threshold', None)
    n_classifiers = int(best_params['n_classifiers'])
    sampling_method = best_params['sampling_method']
    
    classifiers = []
    X_test_samples = []

    for i in range(n_classifiers):
        if sampling_method == 'oob':
            X_sample, y_sample, X_oob, y_oob = create_bootstrap_sample(X_train, y_train, method=sampling_method, random_state=i)
        else:
            X_sample, y_sample = create_bootstrap_sample(X_train, y_train, method=sampling_method, random_state=i)
        
        X_projected = transformer.transform(X_sample)
        X_test_projected = apply_projection_to_test(X_test, transformer)
        
        if sampling_method == 'oob':
            X_oob_projected = apply_projection_to_test(X_oob, transformer)
        
        logger.info(f"Bootstrap sample {i} shape: {X_sample.shape}, Projected shape: {X_projected.shape}")

        knn = CustomKNN(k=int(best_params['n_neighbors']), distance_metric=best_params['metric'], device='cuda')
        knn.fit(X_projected, y_sample)
        classifiers.append(knn)
        X_test_samples.append(X_test_projected)

        torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

    if sampling_method == 'oob':
        evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, variance_threshold, n_classifiers, output_file, best_params, sampling_method, X_oob_projected, y_oob)
    else:
        evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, variance_threshold, n_classifiers, output_file, best_params, sampling_method)
    
    logger.info(f"Completed for method={method}, n_components={n_components}, variance_threshold={variance_threshold}, n_classifiers={n_classifiers}, sampling_method={sampling_method}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Forest-KNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    logger.info("Forest-KNN evaluation completed.")
