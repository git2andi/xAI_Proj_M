import os
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
import pandas as pd
import itertools

root_path = "./database"

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
    
    print(f"Loaded dataset {dataset_name} with shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_bootstrap_samples(X, y, n_samples):
    print(f"Creating {n_samples} bootstrap samples...")
    X_samples, y_samples = [], []
    for i in range(n_samples):
        X_resampled, y_resampled = resample(X, y)
        X_samples.append(X_resampled)
        y_samples.append(y_resampled)
        print(f"  Created bootstrap sample {i+1}/{n_samples}")
    return X_samples, y_samples

def apply_projections(X_samples, method='random', n_components=50):
    print(f"Applying {method} projection with {n_components} components to bootstrap samples...")
    projected_samples = []
    transformers = []
    if method == 'random':
        for i, X in enumerate(X_samples):
            transformer = GaussianRandomProjection(n_components=n_components)
            X_projected = transformer.fit_transform(X)
            projected_samples.append(X_projected)
            transformers.append(transformer)
            print(f"  Applied random projection to sample {i+1}/{len(X_samples)}")
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        for i, X in enumerate(X_samples):
            X_projected = pca.fit_transform(X)
            projected_samples.append(X_projected)
            transformers.append(pca)
            print(f"  Applied PCA to sample {i+1}/{len(X_samples)}")
    return projected_samples, transformers

def apply_projections_to_test(X_test, transformers):
    print(f"Applying projections to test data...")
    X_test_projected_samples = []
    for i, transformer in enumerate(transformers):
        X_test_projected_samples.append(transformer.transform(X_test))
        print(f"  Applied projection {i+1}/{len(transformers)} to test data")
    return X_test_projected_samples

class CustomKNN:
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y):
        self.X_train = torch.tensor(X, device=self.device)
        self.y_train = torch.tensor(y, device=self.device)

    def predict(self, X):
        X = torch.tensor(X, device=self.device)
        distances = torch.cdist(X, self.X_train)
        neighbors = distances.argsort(dim=1)[:, :self.k]
        top_labels = self.y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

def train_knn_models(X_samples, y_samples, k, device='cuda'):
    print(f"Training {len(X_samples)} kNN models with k={k}...")
    classifiers = []
    for i, (X, y) in enumerate(zip(X_samples, y_samples)):
        knn = CustomKNN(k=k, device=device)
        knn.fit(X, y)
        classifiers.append(knn)
        print(f"  Trained kNN model {i+1}/{len(X_samples)} with k={k}")
    return classifiers

def predict_ensemble(classifiers, X_test_samples):
    print(f"Predicting with ensemble of {len(classifiers)} kNN models...")
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test)
        print(f"  Predicted with kNN model {i+1}/{len(classifiers)}")
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions

def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, k, n_classifiers, output_file):
    y_pred = predict_ensemble(classifiers)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}: {accuracy:.4f}")
    
    results = {
        'method': method,
        'n_components': n_components,
        'k': k,
        'n_classifiers': n_classifiers,
        'accuracy': accuracy
    }
    df = pd.DataFrame([results])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Results saved to {output_file}")

def main_forest_knn(dataset_name):
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)
    
    output_file = f'{dataset_name}_forest_knn_results.csv'
    
    methods = ['random', 'pca']
    n_components_list = [50, 100]
    k_values = [5, 10]
    n_classifiers_list = [10, 20]
    
    for method, n_components, k, n_classifiers in itertools.product(methods, n_components_list, k_values, n_classifiers_list):
        print(f"\nStarting evaluation for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}...")
        X_samples, y_samples = create_bootstrap_samples(X_train, y_train, n_classifiers)
        X_projected_samples, transformers = apply_projections(X_samples, method=method, n_components=n_components)
        X_test_projected_samples = apply_projections_to_test(X_test, transformers)
        classifiers = train_knn_models(X_projected_samples, y_samples, k=k, device='cuda')
        evaluate_and_save_results(classifiers, X_test_projected_samples, y_test, method, n_components, k, n_classifiers, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Starting Forest-kNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    print("Forest-kNN evaluation completed.")
