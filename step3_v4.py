import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import itertools
import logging
import random
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit


# Includes Boosting which is wrong lol

root_path = "./database"

def setup_logging(dataset_name):
    """
    Set up logging for the script.
    """
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
    """
    Load embeddings and labels for the specified dataset.
    """
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

def create_bootstrap_sample(X, y, method='bootstrap', random_state=None):
    """
    Create a sample from the training data using various sampling methods.
    """
    if method == 'bootstrap':
        return resample(X, y, random_state=random_state)
    elif method == 'stratified':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
        for train_index, test_index in sss.split(X, y):
            X_resampled, y_resampled = X[train_index], y[train_index]
        return X_resampled, y_resampled
    elif method == 'balanced':
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        X_resampled, y_resampled = [], []
        for class_idx in range(len(class_counts)):
            class_samples = X[y == class_idx]
            resampled_class_samples = resample(class_samples, n_samples=max_count, random_state=random_state)
            X_resampled.append(resampled_class_samples)
            y_resampled.append(np.full(max_count, class_idx))
        return np.vstack(X_resampled), np.hstack(y_resampled)
    elif method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    elif method == 'oob':
        bootstrap_samples = resample(np.arange(len(X)), random_state=random_state)
        oob_samples = np.setdiff1d(np.arange(len(X)), bootstrap_samples)
        return X[bootstrap_samples], y[bootstrap_samples], X[oob_samples], y[oob_samples]
    else:
        raise ValueError("Unknown sampling method.")


def apply_projection(X, method='random', n_components=50, random_state=None):
    """
    Apply dimensionality reduction to the data.
    """
    if method == 'random':
        transformer = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    elif method == 'pca':
        transformer = PCA(n_components=n_components, random_state=random_state)
    X_projected = transformer.fit_transform(X)
    return X_projected, transformer

def apply_projection_to_test(X_test, transformer):
    """
    Apply the same dimensionality reduction to the test data.
    """
    return transformer.transform(X_test)

class CustomKNN:
    """
    Custom k-Nearest Neighbors (kNN) classifier with GPU support.
    """
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y):
        """
        Fit the kNN model using the training data.
        """
        self.X_train = torch.tensor(X, device=self.device)
        self.y_train = torch.tensor(y, device=self.device)

    def predict(self, X, batch_size=1000):
        """
        Predict the labels for the input data using kNN.
        """
        X = torch.tensor(X, device=self.device)
        num_samples = X.shape[0]
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            distances = torch.cdist(batch, self.X_train)
            neighbors = distances.argsort(dim=1)[:, :self.k]
            top_labels = self.y_train[neighbors]
            batch_predictions = torch.mode(top_labels, dim=1).values
            predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

def bayesian_optimization_knn(X_train, y_train):
    """
    Perform Bayesian Optimization to find the best kNN parameters.
    """
    def knn_evaluate(n_neighbors, weights_idx, metric_idx):
        n_neighbors = int(n_neighbors)
        weights_options = ['uniform', 'distance']
        metric_options = ['euclidean', 'manhattan', 'chebyshev']
        weights = weights_options[int(weights_idx)]
        metric = metric_options[int(metric_idx)]
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        knn.fit(X_train, y_train)
        score = knn.score(X_train, y_train)
        return score
    
    pbounds = {
        'n_neighbors': (10, 20),
        'weights_idx': (0, 1),
        'metric_idx': (0, 2)
    }
    
    optimizer = BayesianOptimization(
        f=knn_evaluate,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=10, n_iter=30)
    
    best_params = optimizer.max['params']
    best_params['n_neighbors'] = int(best_params['n_neighbors'])
    best_params['weights'] = ['uniform', 'distance'][int(best_params.pop('weights_idx'))]
    best_params['metric'] = ['euclidean', 'manhattan', 'cosine'][int(best_params.pop('metric_idx'))]
    
    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {best_params}")
    
    return best_knn, best_params


def train_boosted_knn(X_train, y_train):
    """
    Train a boosted kNN model.
    """
    boost_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    boost_model.fit(X_train, y_train)
    return boost_model

def predict_with_ensemble(classifiers, X_test_samples):
    """
    Make predictions using an ensemble of classifiers.
    """
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions



def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, n_classifiers, output_file, best_params, sampling_method, X_oob=None, y_oob=None):
    """
    Evaluate the ensemble and save results.
    """
    y_pred = predict_with_ensemble(classifiers, X_test_samples)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    k = best_params['n_neighbors']
    boosting_technique = "GradientBoostingClassifier"
    
    logger.info(f"Accuracy for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}, sampling_method={sampling_method}: {accuracy:.4f}")
    
    results = {
        'method': method,
        'n_components': n_components,
        'k': k,
        'n_classifiers': n_classifiers,
        'best_params': best_params,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'sampling_method': sampling_method,
        'boosting_technique': boosting_technique,
        'total_misclassified': len(misclassified_indices),
        'misclassified_indices': random.sample(list(misclassified_indices), 5) if len(misclassified_indices) > 0 else 'None'
    }

    # Ensure this header is used only if the file doesn't exist yet
    header = not os.path.exists(output_file)
    df = pd.DataFrame([results])
    df.to_csv(output_file, mode='a', header=header, index=False)
    logger.info(f"Results saved to {output_file}")




def main_forest_knn(dataset_name):
    """
    Main function to run Forest-kNN.
    """
    global logger
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Forest-kNN on dataset {dataset_name}")
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    
    output_file = f'{dataset_name}_forest_knn_results.csv'
    
    methods = ['random', 'pca']
    n_components_list = [20, 50, 100]
    n_classifiers_list = [10, 20, 30]
    sampling_methods = ['bootstrap', 'stratified', 'balanced', 'smote', 'oob']
    
    for method, n_components, n_classifiers, sampling_method in itertools.product(methods, n_components_list, n_classifiers_list, sampling_methods):
        logger.info(f"\nStarting evaluation for method={method}, n_components={n_components}, n_classifiers={n_classifiers}, sampling_method={sampling_method}...")
        classifiers = []
        X_test_samples = []
        
        for i in range(n_classifiers):
            if sampling_method == 'oob':
                X_sample, y_sample, X_oob, y_oob = create_bootstrap_sample(X_train, y_train, method=sampling_method, random_state=i)
            else:
                X_sample, y_sample = create_bootstrap_sample(X_train, y_train, method=sampling_method, random_state=i)
            
            X_projected, transformer = apply_projection(X_sample, method=method, n_components=n_components, random_state=i)
            X_test_projected = apply_projection_to_test(X_test, transformer)
            
            if sampling_method == 'oob':
                X_oob_projected = apply_projection_to_test(X_oob, transformer)
            
            logger.info(f"Bootstrap sample {i} shape: {X_sample.shape}, Projected shape: {X_projected.shape}")
            
            # Perform Bayesian optimization for the best kNN parameters
            best_knn, best_params = bayesian_optimization_knn(X_projected, y_sample)
            # Train boosted kNN model
            boosted_knn = train_boosted_knn(X_projected, y_sample)
            classifiers.append(boosted_knn)
            X_test_samples.append(X_test_projected)
        
        if sampling_method == 'oob':
            evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, n_classifiers, output_file, best_params, sampling_method, X_oob_projected, y_oob)
        else:
            evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, n_classifiers, output_file, best_params, sampling_method)
        logger.info(f"Evaluation completed for method={method}, n_components={n_components}, n_classifiers={n_classifiers}, sampling_method={sampling_method}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Forest-kNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    logger.info("Forest-kNN evaluation completed.")
