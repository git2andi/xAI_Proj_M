import os
import numpy as np
import torch
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

root_path = "./database"
output_file = 'forest_knn_results.csv'
train_acc_file = 'forest_knn_train_accuracies.csv'

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def load_embeddings_and_labels(dataset_name, root_path):
    print(f"Loading embeddings and labels for {dataset_name}...")
    dataset_path = os.path.join(root_path, dataset_name)
    
    train_data = np.load(os.path.join(dataset_path, 'train.npz'))
    X_train, y_train = train_data['embeddings'], train_data['labels'].reshape(-1,)
    
    if os.path.exists(os.path.join(dataset_path, 'val.npz')):
        val_data = np.load(os.path.join(dataset_path, 'val.npz'))
        X_val, y_val = val_data['embeddings'], val_data['labels'].reshape(-1,)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    
    test_data = np.load(os.path.join(dataset_path, 'test.npz'))
    X_test, y_test = test_data['embeddings'], test_data['labels'].reshape(-1,)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def select_important_features(X, y, n_features=100):
    selector = SelectKBest(mutual_info_classif, k=n_features)
    return selector.fit_transform(X, y)

def augment_features(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    return poly.fit_transform(X)

def normalize_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm

def create_bootstrap_samples(X, y, n_samples):
    X_samples, y_samples = [], []
    for _ in tqdm(range(n_samples)):
        X_resampled, y_resampled = resample(X, y, random_state=None)
        X_samples.append(X_resampled)
        y_samples.append(y_resampled)
    return X_samples, y_samples

def apply_projections(X_samples, method='random', n_components=50):
    projections = []
    transformers = []
    for X in tqdm(X_samples):
        if method == 'random':
            transformer = GaussianRandomProjection(n_components=n_components, random_state=random.randint(0, 10000))
        elif method == 'pca':
            transformer = PCA(n_components=n_components)
        elif method == 'kernel_pca':
            transformer = KernelPCA(n_components=n_components, kernel='rbf')
        X_projected = transformer.fit_transform(X)
        projections.append(X_projected)
        transformers.append(transformer)
    return projections, transformers

def apply_projections_to_test(X_test, transformers):
    return [transformer.transform(X_test) for transformer in tqdm(transformers)]

class CustomKNN:
    def __init__(self, k=5, device='cuda'):
        self.k = k
        self.device = device

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train = torch.tensor(X, device=self.device, dtype=torch.float)
        self.y_train = torch.tensor(y, device=self.device, dtype=torch.long)
        return self

    def predict(self, X):
        X = torch.tensor(X, device=self.device, dtype=torch.float)
        distances = torch.cdist(X, self.X_train)
        neighbors = distances.argsort(dim=1)[:, :self.k]
        top_labels = self.y_train[neighbors]
        predictions = torch.mode(top_labels, dim=1).values
        return predictions.cpu().numpy()

def train_knn_models(X_samples, y_samples, X_val, y_val, k=5, device='cuda'):
    classifiers = []
    train_accuracies = []
    val_accuracies = []
    
    for i, (X, y) in enumerate(zip(X_samples, y_samples)):
        knn = CustomKNN(k=k, device=device)
        knn.fit(X, y)
        classifiers.append(knn)
        
        y_train_pred = knn.predict(X)
        train_accuracy = accuracy_score(y, y_train_pred)
        train_accuracies.append(train_accuracy)
        
        y_val_pred = knn.predict(X_val[i])
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_accuracy)
    
    return classifiers, train_accuracies, val_accuracies

def train_boosted_knn(X_train, y_train, X_val, y_val, transformers, k=5, n_estimators=50):
    boosted_knn_models = []
    boosted_weights = []

    for n in range(n_estimators):
        X_train_projected = [transformer.transform(X_train) for transformer in transformers]
        X_val_projected = [transformer.transform(X_val) for transformer in transformers]

        knn_models = []
        for X_proj in X_train_projected:
            knn = CustomKNN(k=k, device='cuda')
            knn.fit(X_proj, y_train)
            knn_models.append(knn)

        val_accuracies = []
        for knn, X_proj in zip(knn_models, X_val_projected):
            y_val_pred = knn.predict(X_proj)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_accuracies.append(val_accuracy)
        
        best_knn = knn_models[np.argmax(val_accuracies)]
        best_val_accuracy = max(val_accuracies)
        weight = best_val_accuracy / sum(val_accuracies)
        
        boosted_knn_models.append(best_knn)
        boosted_weights.append(weight)
    
    return boosted_knn_models, boosted_weights

def predict_ensemble(classifiers, X_test_samples):
    predictions = np.zeros((X_test_samples[0].shape[0], len(classifiers)))
    for i, (clf, X_test) in enumerate(zip(classifiers, X_test_samples)):
        predictions[:, i] = clf.predict(X_test)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
    return final_predictions

def predict_weighted_ensemble(models, X_samples, weights):
    predictions = np.zeros((X_samples[0].shape[0], len(models)))
    for i, (model, X) in enumerate(zip(models, X_samples)):
        predictions[:, i] = model.predict(X)
    
    weighted_votes = np.zeros(predictions.shape[0])
    for i, weight in enumerate(weights):
        weighted_votes += weight * predictions[:, i]
    final_predictions = np.round(weighted_votes / sum(weights)).astype(int)
    return final_predictions

def evaluate_and_save_results(classifiers, X_test_samples, y_test, method, n_components, k, dataset_name, output_file, train_acc_file, train_accuracies, weights=None):
    print(f"Evaluating results for method={method}, n_components={n_components}, k={k}...")
    if weights:
        y_pred = predict_weighted_ensemble(classifiers, X_test_samples, weights)
    else:
        y_pred = predict_ensemble(classifiers, X_test_samples)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    result = {
        'dataset': dataset_name,
        'method': method,
        'n_components': n_components,
        'k': k,
        'accuracy': accuracy,
        'confusion_matrix': str(conf_matrix.tolist())  # Convert to string for CSV serialization
    }
    df = pd.DataFrame([result])
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Results saved to {output_file}")

    # Save training accuracies
    train_acc_df = pd.DataFrame({'dataset': [dataset_name] * len(train_accuracies), 'method': [method] * len(train_accuracies), 'train_accuracy': train_accuracies})
    train_acc_df.to_csv(train_acc_file, mode='a', header=not os.path.exists(train_acc_file), index=False)
    print(f"Training accuracies saved to {train_acc_file}")

def main_forest_knn(dataset_name):
    X_train, y_train, X_val, y_val, X_test, y_test = load_embeddings_and_labels(dataset_name, root_path)

    n_classifiers_values = [20]
    n_components_list = [50]
    k_values = [5]
    methods = ['random', 'pca', 'kernel_pca']

    for n_classifiers in n_classifiers_values:
        X_train_selected = select_important_features(X_train, y_train)
        X_train_augmented = augment_features(X_train_selected)
        X_val_selected = select_important_features(X_val, y_val)
        X_val_augmented = augment_features(X_val_selected)
        X_test_selected = select_important_features(X_test, y_test)
        X_test_augmented = augment_features(X_test_selected)
        X_train_norm, X_val_norm, X_test_norm = normalize_features(X_train_augmented, X_val_augmented, X_test_augmented)

        X_samples, y_samples = create_bootstrap_samples(X_train_norm, y_train, n_classifiers)
        
        for method in methods:
            for n_components in n_components_list:
                for k in k_values:
                    print(f"\nStarting evaluation for method={method}, n_components={n_components}, k={k}, n_classifiers={n_classifiers}...")
                    X_projected_samples, transformers = apply_projections(X_samples, method=method, n_components=n_components)
                    X_val_projected = apply_projections_to_test(X_val_norm, transformers)
                    X_test_projected = apply_projections_to_test(X_test_norm, transformers)
                    
                    try:
                        classifiers, train_accuracies, val_accuracies = train_knn_models(X_projected_samples, y_samples, X_val_projected, y_val, k=k, device='cuda')
                    except torch.cuda.OutOfMemoryError:
                        print("CUDA out of memory. Switching to CPU.")
                        classifiers, train_accuracies, val_accuracies = train_knn_models(X_projected_samples, y_samples, X_val_projected, y_val, k=k, device='cpu')
                    
                    evaluate_and_save_results(classifiers, X_test_projected, y_test, method, n_components, k, dataset_name, output_file, train_acc_file, train_accuracies, weights=None)
                    
                    boosted_knn_models, boosted_weights = train_boosted_knn(X_train_norm, y_train, X_val_norm, y_val, transformers, k=k)
                    evaluate_and_save_results(boosted_knn_models, X_test_projected, y_test, method, n_components, k, dataset_name, output_file, train_acc_file, train_accuracies, weights=boosted_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forest-kNN Classification with PyTorch")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100", "dermamnist", "breastmnist"], help="Dataset to use")
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Starting Forest-kNN on dataset {dataset_name}")
    main_forest_knn(dataset_name)
    print("Forest-kNN evaluation completed.")
