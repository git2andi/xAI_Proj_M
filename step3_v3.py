import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
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

def create_bootstrap_sample(X, y, random_state=None):
    """
    Create a bootstrap sample from the training data.
    """
    return resample(X, y, random_state=random_state)

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

def grid_search_knn(X_train, y_train):
    """
    Perform grid search to find the best kNN parameters.
    """
    param_grid = {
        'n_neighbors': [10, 15, 17, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation accuracy: {grid_search.best_score_}")

    return grid_search.best_estimator_, grid_search.best_params_

def train_boosted_knn(X_train, y_train):
    """
    Train a boosted kNN model.
    """
    boost_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    boost_model.fit(X_train, y_train)
    return boost_model

def predict_with_ensemble(classifiers, X_test_samples, batch_size=1000):
    """
    Make predictions using an ensemble of classifiers.
    """
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions

def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, k, n_classifiers, output_file, best_params):
    """
    Evaluate the ensemble and save results.
    """
    y_pred = predict_with_ensemble(classifiers, X_test_samples)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    logger.info(f"Accuracy for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}: {accuracy:.4f}")
    
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
        'total_misclassified': len(misclassified_indices),
        'misclassified_indices': random.sample(list(misclassified_indices), 5) if len(misclassified_indices) > 0 else 'None'
    }
    df = pd.DataFrame([results])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
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
    n_components_list = [50, 100, 200]
    k_values = [10, 15, 17, 20]
    n_classifiers_list = [20, 30, 50]
    
    for method, n_components, n_classifiers in itertools.product(methods, n_components_list, n_classifiers_list):
        logger.info(f"\nStarting evaluation for method={method}, n_components={n_components}, n_classifiers={n_classifiers}...")
        classifiers = []
        X_test_samples = []
        
        for i in range(n_classifiers):
            X_sample, y_sample = create_bootstrap_sample(X_train, y_train, random_state=i)
            X_projected, transformer = apply_projection(X_sample, method=method, n_components=n_components, random_state=i)
            X_test_projected = apply_projection_to_test(X_test, transformer)
            
            logger.info(f"Bootstrap sample {i} shape: {X_sample.shape}, Projected shape: {X_projected.shape}")
            
            # Perform grid search for the best kNN parameters
            best_knn, best_params = grid_search_knn(X_projected, y_sample)
            # Train boosted kNN model
            boosted_knn = train_boosted_knn(X_projected, y_sample)
            classifiers.append(boosted_knn)
            X_test_samples.append(X_test_projected)
        
        evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, None, n_classifiers, output_file, best_params)
        logger.info(f"Evaluation completed for method={method}, n_components={n_components}, n_classifiers={n_classifiers}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Forest-kNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    logger.info("Forest-kNN evaluation completed.")
